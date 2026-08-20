"""Thin Python inference loop for the Rust-owned rollout protocol."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import warnings
from time import perf_counter


@dataclass(frozen=True, slots=True)
class NativeInferenceStats:
    requests: int
    rows: int
    useful_tokens: int
    padded_tokens: int
    useful_actions: int
    padded_actions: int
    compile_fallbacks: int
    inference_seconds: float
    native_seconds: float
    automatic_rows: int


class NativeInferenceRunner:
    """Execute policy inference while native code owns rollout state.

    Sampling deliberately remains outside compiled actor forwards.  The
    request already has a deterministic native row order, so a seeded torch
    generator sees the same categorical segments regardless of Rust worker
    timing.
    """

    def __init__(
        self,
        policies,
        *,
        device="cpu",
        backend="sdpa",
        use_bf16=False,
        generator=None,
        deterministic_policy_slots=(),
        compile_cuda=False,
        require_state_values=True,
        gae_lambda=1.0,
        profiler=None,
    ):
        import torch

        self.policies = dict(policies)
        self.device = torch.device(device)
        self.backend = str(backend)
        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda")
        self.generator = generator
        self.deterministic_policy_slots = frozenset(
            map(int, deterministic_policy_slots)
        )
        self.compile_cuda = bool(compile_cuda and self.device.type == "cuda")
        self.require_state_values = bool(require_state_values)
        self.gae_lambda = float(gae_lambda)
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("GAE lambda must be in [0, 1]")
        self.profiler = profiler
        self._compiled = {}
        self._compile_fallbacks = 0
        self._bucket_counts = defaultdict(int)
        self._inference_seconds = 0.0
        self._native_seconds = 0.0

    @staticmethod
    def _tensor(torch, array, *, device):
        # NumPy views are intentionally read-only. Torch never mutates model
        # inputs, so sharing them on CPU is safe; suppress only its generic
        # writeability warning. CUDA transfer creates device-owned storage.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            value = torch.as_tensor(array)
        return value.to(device=device, non_blocking=device.type == "cuda")

    def _inputs(self, request):
        import torch

        values = request.model_inputs(backend=self.backend)
        return {
            name: (
                value
                if name == "backend"
                else self._tensor(torch, value, device=self.device)
            )
            for name, value in values.items()
        }

    def _forward(self, policy, request, inputs):
        key = (
            int(request.policy_slot),
            int(request.sequence_bucket),
            int(request.action_bucket),
        )
        if not self.compile_cuda:
            return policy.forward_actor(
                **inputs,
                compute_entropy=False,
                compute_value=True,
                compute_auxiliary=False,
            )
        self._bucket_counts[key] += 1
        if self._bucket_counts[key] < 8:
            return policy.forward_actor(
                **inputs,
                compute_entropy=False,
                compute_value=True,
                compute_auxiliary=False,
            )
        forward = self._compiled.get(key)
        if forward is None:
            import torch

            try:
                forward = torch.compile(
                    policy.forward_actor,
                    mode="default",
                    dynamic=True,
                )
            except Exception:
                self._compile_fallbacks += 1
                forward = False
            self._compiled[key] = forward
        if forward is False:
            return policy.forward_actor(
                **inputs,
                compute_entropy=False,
                compute_value=True,
                compute_auxiliary=False,
            )
        try:
            return forward(
                **inputs,
                compute_entropy=False,
                compute_value=True,
                compute_auxiliary=False,
            )
        except Exception:
            # A bucket that cannot compile remains eager for the rest of the
            # process, avoiding repeated graph breaks in the rollout hot path.
            self._compile_fallbacks += 1
            self._compiled[key] = False
            return policy.forward_actor(
                **inputs,
                compute_entropy=False,
                compute_value=True,
                compute_auxiliary=False,
            )

    def infer(self, request):
        import torch

        try:
            policy = self.policies[int(request.policy_slot)]
        except KeyError as error:
            raise ValueError(
                f"no policy is registered for native slot {request.policy_slot}"
            ) from error
        if hasattr(policy, "eval"):
            policy.eval()
        inputs = self._inputs(request)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self._forward(policy, request, inputs)
            selected_global = policy.sample(
                output,
                inputs["action_offsets"],
                generator=self.generator,
                deterministic=(
                    int(request.policy_slot)
                    in self.deterministic_policy_slots
                ),
            )
            starts = inputs["action_offsets"][:-1]
            selected_groups = selected_global - starts
            selected_logp = output.log_probabilities.index_select(
                0, selected_global
            )
            state_values = getattr(output, "state_values", None)
            if state_values is None:
                if self.require_state_values:
                    raise RuntimeError(
                        "rollout policy did not produce state values"
                    )
                state_values = selected_logp.new_zeros(selected_groups.shape)
            packed = torch.stack(
                (
                    selected_groups.float(),
                    selected_logp.float(),
                    state_values.float(),
                ),
                dim=1,
            ).cpu()
        return (
            packed[:, 0].to(dtype=torch.int64).tolist(),
            packed[:, 1].tolist(),
            packed[:, 2].tolist(),
        )

    def infer_seeded(self, request, row_seeds):
        """Inference with one batch-size-invariant sampling seed per row."""
        import torch
        from ..encoding.actions import segmented_sample

        row_seeds = tuple(map(int, row_seeds))
        if len(row_seeds) != int(request.row_count):
            raise ValueError("one evaluation seed is required per request row")
        try:
            policy = self.policies[int(request.policy_slot)]
        except KeyError as error:
            raise ValueError(
                f"no policy is registered for native slot {request.policy_slot}"
            ) from error
        if hasattr(policy, "eval"):
            policy.eval()
        inputs = self._inputs(request)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self._forward(policy, request, inputs)
        state_values = getattr(output, "state_values", None)
        if state_values is None:
            if self.require_state_values:
                raise RuntimeError("rollout policy did not produce state values")
            state_values = output.log_probabilities.new_zeros(
                int(request.row_count),
            )
        host_logp = output.log_probabilities.detach().float().cpu()
        offsets = inputs["action_offsets"].detach().cpu().tolist()
        deterministic = (
            int(request.policy_slot) in self.deterministic_policy_slots
        )
        selected = []
        old_logp = []
        for row, seed in enumerate(row_seeds):
            start, end = offsets[row:row + 2]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            local = int(segmented_sample(
                host_logp[start:end],
                torch.tensor([0, end - start], dtype=torch.long),
                generator=generator,
                deterministic=deterministic,
            )[0])
            selected.append(local)
            old_logp.append(float(host_logp[start + local]))
        return selected, old_logp, state_values.detach().float().cpu().tolist()

    def run_chunk(self, engine):
        """Drive an initialized, lineup-registered engine to completion."""
        while not engine.complete:
            native_started = perf_counter()
            request = engine.next_request()
            self._native_seconds += perf_counter() - native_started
            if request is None:
                break
            inference_started = perf_counter()
            selected, old_logp, old_state_values = self.infer(request)
            self._inference_seconds += perf_counter() - inference_started
            native_started = perf_counter()
            engine.submit(
                request.request_id, selected, old_logp, old_state_values,
            )
            self._native_seconds += perf_counter() - native_started
        if not engine.complete:
            raise RuntimeError("native rollout scheduler stopped before completion")
        return engine.take_chunk()

    def prepare_training_chunk(self, chunk):
        """Deduplicate boundary critic work and finish native PPO targets."""
        import numpy as np
        import torch

        boundary = chunk.boundary_batch()
        group_ids = np.asarray(boundary["boundary_group_ids"])
        policy_slots = np.asarray(boundary["policy_slots"])
        values = np.empty(len(group_ids), dtype=np.float32)
        for slot in sorted(set(map(int, policy_slots))):
            try:
                policy = self.policies[slot]
            except KeyError as error:
                raise ValueError(
                    f"no policy is registered for native slot {slot}"
                ) from error
            indices = np.flatnonzero(policy_slots == slot)
            inputs = {
                "decision_seats": self._tensor(
                    torch,
                    np.asarray(boundary["decision_seats"])[indices],
                    device=self.device,
                ),
                "rank_boundary_features": self._tensor(
                    torch,
                    np.asarray(boundary["rank_boundary_features"])[indices],
                    device=self.device,
                ),
            }
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                output = policy.forward_critic(**inputs)
            values[indices] = output.rank_values.float().cpu().numpy()
        chunk.set_boundary_values(group_ids.tolist(), values.tolist())
        chunk.finish_targets(self.gae_lambda)
        return chunk

    def stats(self, engine) -> NativeInferenceStats:
        metrics = engine.metrics()
        return NativeInferenceStats(
            requests=int(metrics["inference_requests"]),
            rows=int(metrics["inference_rows"]),
            useful_tokens=int(metrics["useful_tokens"]),
            padded_tokens=int(metrics["padded_tokens"]),
            useful_actions=int(metrics["useful_actions"]),
            padded_actions=int(metrics["padded_actions"]),
            compile_fallbacks=self._compile_fallbacks,
            inference_seconds=self._inference_seconds,
            native_seconds=self._native_seconds,
            automatic_rows=int(metrics["automatic_rows"]),
        )
