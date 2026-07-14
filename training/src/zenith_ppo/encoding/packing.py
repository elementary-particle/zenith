"""Flat factor storage and deterministic token-budget packing."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import encode_native_actions
from .events import encode_history
from .schema import Segment, TokenKind
from .state import encode_critic_factors, encode_state_factors
from ..env.projection import project_decision
from ..types import DecisionBinding


_ACTOR_QUERY_FACTORS = None
_ZERO_NUMERIC_ROW = None


def _query_rows():
    global _ACTOR_QUERY_FACTORS, _ZERO_NUMERIC_ROW
    if _ACTOR_QUERY_FACTORS is None:
        import numpy as np
        _ACTOR_QUERY_FACTORS = np.asarray([(
            Segment.ACTOR_QUERY, TokenKind.QUERY, 1, 1, 0, 0, 0, 0, 0, 0
        )], dtype=np.uint8)
        _ZERO_NUMERIC_ROW = np.zeros((1, 8), dtype=np.float32)
    return _ACTOR_QUERY_FACTORS, _ZERO_NUMERIC_ROW


@dataclass(frozen=True, slots=True)
class PackedBatch:
    order: tuple[int, ...]
    batches: tuple[tuple[int, ...], ...]
    total_tokens: int
    padded_tokens: int


@dataclass(frozen=True, slots=True, eq=False)
class EncodedDecision:
    binding: DecisionBinding
    token_factors: object
    token_numeric: object
    actor_query_index: int
    critic_factors: object
    critic_numeric: object
    opponent_count_targets: object
    opponent_tenpai_targets: object
    action_factors: object
    action_representatives: tuple[int, ...]
    native_actions: tuple[object, ...]

    def __eq__(self, other):
        if not isinstance(other, EncodedDecision):
            return NotImplemented
        import numpy as np
        return (
            self.binding == other.binding
            and np.array_equal(self.token_factors, other.token_factors)
            and np.array_equal(self.token_numeric, other.token_numeric)
            and self.actor_query_index == other.actor_query_index
            and np.array_equal(self.critic_factors, other.critic_factors)
            and np.array_equal(self.critic_numeric, other.critic_numeric)
            and np.array_equal(self.opponent_count_targets, other.opponent_count_targets)
            and np.array_equal(self.opponent_tenpai_targets, other.opponent_tenpai_targets)
            and np.array_equal(self.action_factors, other.action_factors)
            and self.action_representatives == other.action_representatives
            and self.native_actions == other.native_actions
        )


def opponent_count_targets(hidden_counts, *, observer: int):
    """Return concealed-count labels in relative-seat order 2, 3, 4."""
    import numpy as np

    return np.asarray([
        list(hidden_counts[(int(observer) + relative - 1) % 4])
        for relative in (2, 3, 4)
    ], dtype=np.uint8)


def opponent_tenpai_targets(counts, open_melds):
    """Batch and deduplicate native shanten analysis for ephemeral tenpai labels."""
    import numpy as np
    from ..env.analysis import analyze, unique_rows

    rows, melds, inverse = unique_rows(counts, open_melds)
    if not rows:
        return np.zeros(0, dtype=np.float32)
    analysis = analyze(rows, melds)
    return np.asarray(
        [int(analysis.shanten[index, 0]) == 0 for index in inverse], dtype=np.float32
    )


def mean_token_length(packed: PackedBatch, sequence_count: int) -> float:
    if int(sequence_count) <= 0:
        return 0.0
    return packed.total_tokens / int(sequence_count)


def padding_fraction(packed: PackedBatch) -> float:
    """Fraction of materialized token slots occupied by padding, in [0, 1]."""
    if packed.padded_tokens < packed.total_tokens:
        raise ValueError("padded token count cannot be smaller than useful tokens")
    if not packed.padded_tokens:
        return 0.0
    return (packed.padded_tokens - packed.total_tokens) / packed.padded_tokens


def encode_native_batch(batch, histories, *, critic_mode="privileged", event_cache=None):
    import numpy as np

    actor_query_factors, zero_numeric_row = _query_rows()
    encoded = []
    tenpai_counts, tenpai_melds = [], []
    for state in batch.transition.states:
        for decision in state.decisions:
            observer = int(decision.seat)
            binding = DecisionBinding(
                decision.environment_id,
                decision.episode_generation,
                decision.frame_id,
                observer,
            )
            store = histories.get(binding.environment_id, binding.episode_generation)
            if event_cache is None:
                event_tokens = encode_history(
                    store.rows,
                    observer=observer,
                    generation=binding.episode_generation,
                )
                event_factors = np.asarray(
                    [token.categorical() for token in event_tokens], dtype=np.uint8
                ).reshape(-1, 10)
                from .schema import numeric_features
                event_numeric = np.asarray(
                    [numeric_features(token) for token in event_tokens], dtype=np.float32
                ).reshape(-1, 8)
            else:
                event_prefix = event_cache.encode(store, observer=observer)
                event_factors = event_prefix.categorical_array
                event_numeric = event_prefix.numeric_array
            frame, actor, critic = project_decision(
                state,
                decision,
                observer=observer,
                critic_mode=critic_mode,
            )
            actor_factors, actor_numeric = encode_state_factors(
                frame,
                actor,
                observer=observer,
            )
            actor_query = len(event_factors) + len(actor_factors)
            critic_frame = dict(frame)
            critic_frame.update(critic)
            critic_factors, critic_numeric = encode_critic_factors(
                critic_frame, observer=observer
            )
            actions = encode_native_actions(decision.actions, observer=observer)
            token_factors = np.concatenate((event_factors, actor_factors, actor_query_factors), axis=0)
            token_numeric = np.concatenate((event_numeric, actor_numeric, zero_numeric_row), axis=0)
            if state.hidden is None:
                count_targets = np.zeros((3, 34), dtype=np.uint8)
                tenpai_start = -1
            else:
                count_targets = opponent_count_targets(
                    state.hidden.concealed_counts, observer=observer
                )
                open_by_seat = [sum(int(meld.seat) == seat for meld in state.melds)
                                for seat in range(4)]
                tenpai_start = len(tenpai_counts)
                for relative, target in zip((2, 3, 4), count_targets, strict=True):
                    tenpai_counts.append(target)
                    tenpai_melds.append(open_by_seat[(observer + relative - 1) % 4])
            encoded.append(
                EncodedDecision(
                    binding,
                    token_factors,
                    token_numeric,
                    actor_query,
                    critic_factors,
                    critic_numeric,
                    count_targets,
                    tenpai_start,
                    np.asarray(actions.factors, dtype=np.uint8).reshape(-1, 15),
                    actions.representatives,
                    tuple(decision.actions),
                )
            )
    if tenpai_counts:
        from dataclasses import replace
        labels = opponent_tenpai_targets(tenpai_counts, tenpai_melds)
        encoded = [replace(row, opponent_tenpai_targets=(
            labels[row.opponent_tenpai_targets:row.opponent_tenpai_targets + 3]
            if row.opponent_tenpai_targets >= 0 else np.zeros(3, dtype=np.float32)
        )) for row in encoded]
    else:
        encoded = [__import__("dataclasses").replace(
            row, opponent_tenpai_targets=np.zeros(3, dtype=np.float32)) for row in encoded]
    return tuple(encoded)


def model_batch(encoded, *, device="cpu", backend="sdpa"):
    import numpy as np
    import torch

    if not encoded:
        raise ValueError("cannot build an empty model batch")
    lengths_array = np.fromiter(
        (len(row.token_factors) for row in encoded), dtype=np.int64, count=len(encoded)
    )
    maximum = int(lengths_array.max())
    critic_lengths_array = np.fromiter(
        (len(getattr(row, "critic_factors", ())) for row in encoded),
        dtype=np.int64, count=len(encoded)
    )
    critic_maximum = int(critic_lengths_array.max(initial=0))
    pin_memory = torch.device(device).type == "cuda"

    def host_tensor(shape, dtype, *, zero=False):
        tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
        return tensor.zero_() if zero else tensor

    token_factors_host = host_tensor(
        (len(encoded), maximum, 10), torch.long, zero=True
    )
    token_numeric_host = host_tensor(
        (len(encoded), maximum, 8), torch.float32, zero=True
    )
    token_factors_array = token_factors_host.numpy()
    token_numeric_array = token_numeric_host.numpy()
    critic_factors_host = host_tensor((len(encoded), critic_maximum, 10), torch.long, zero=True)
    critic_numeric_host = host_tensor((len(encoded), critic_maximum, 8), torch.float32, zero=True)
    critic_factors_array = critic_factors_host.numpy()
    critic_numeric_array = critic_numeric_host.numpy()
    action_count = sum(len(row.action_factors) for row in encoded)
    action_factors_host = host_tensor((action_count, 15), torch.long)
    action_factors_array = action_factors_host.numpy()
    offsets = [0]
    action_end = 0
    for index, row in enumerate(encoded):
        length = len(row.token_factors)
        token_factors_array[index, :length] = row.token_factors
        token_numeric_array[index, :length] = row.token_numeric
        critic_length = len(getattr(row, "critic_factors", ()))
        if critic_length:
            critic_factors_array[index, :critic_length] = row.critic_factors
            critic_numeric_array[index, :critic_length] = row.critic_numeric
        next_action_end = action_end + len(row.action_factors)
        action_factors_array[action_end:next_action_end] = row.action_factors
        action_end = next_action_end
        offsets.append(action_end)
    lengths_host = host_tensor((len(encoded),), torch.long)
    lengths_host.numpy()[:] = lengths_array
    actor_queries_host = host_tensor((len(encoded),), torch.long)
    actor_queries_host.numpy()[:] = np.fromiter(
        (row.actor_query_index for row in encoded), dtype=np.int64, count=len(encoded)
    )
    critic_lengths_host = host_tensor((len(encoded),), torch.long)
    critic_lengths_host.numpy()[:] = critic_lengths_array
    action_offsets_host = host_tensor((len(offsets),), torch.long)
    action_offsets_host.numpy()[:] = offsets

    def transfer(tensor):
        return tensor.to(device=device, non_blocking=pin_memory)

    return {
        "token_factors": transfer(token_factors_host),
        "token_numeric": transfer(token_numeric_host),
        "critic_factors": transfer(critic_factors_host),
        "critic_numeric": transfer(critic_numeric_host),
        "action_factors": transfer(action_factors_host),
        "actor_query_indices": transfer(actor_queries_host),
        "critic_lengths": transfer(critic_lengths_host),
        "action_offsets": transfer(action_offsets_host),
        "lengths": transfer(lengths_host),
        "backend": backend,
    }


def pack(lengths, token_budget: int, *, sort_window: int = 128,
         max_padding_fraction: float | None = None) -> PackedBatch:
    if token_budget <= 0 or any(length <= 0 or length > token_budget for length in lengths):
        raise ValueError("every sequence must fit the positive token budget")
    if max_padding_fraction is not None and not 0 <= max_padding_fraction < 1:
        raise ValueError("maximum padding fraction must be in [0, 1)")
    if max_padding_fraction is not None:
        # A hard global waste bound and local sort windows interact badly at
        # every window boundary. Global deterministic sort keeps neighboring
        # lengths homogeneous while token_budget still bounds each minibatch.
        order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    else:
        order = []
        for start in range(0, len(lengths), sort_window):
            order.extend(sorted(range(start, min(start + sort_window, len(lengths))),
                                key=lambda index: (-lengths[index], index)))
    batches, current, current_max, current_sum = [], [], 0, 0
    padded = 0
    for index in order:
        next_max = max(current_max, lengths[index])
        next_padded = next_max * (len(current) + 1)
        next_sum = current_sum + lengths[index]
        next_padding = (next_padded - next_sum) / next_padded
        if current and (
            next_padded > token_budget
            or (
                max_padding_fraction is not None
                and next_padding > max_padding_fraction
            )
        ):
            batches.append(tuple(current)); padded += current_max * len(current)
            current, current_max, current_sum = [], 0, 0
        current.append(index)
        current_max = max(current_max, lengths[index])
        current_sum += lengths[index]
    if current:
        batches.append(tuple(current)); padded += current_max * len(current)
    return PackedBatch(tuple(order), tuple(batches), sum(lengths), padded)
