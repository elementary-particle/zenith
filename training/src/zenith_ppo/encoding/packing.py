"""Flat factor storage and deterministic token-budget packing."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import encode_native_actions
from .critic import BOUNDARY_RANK_FEATURES, encode_rank_boundary
from .events import encode_history
from .schema import Segment, TokenKind
from .state import (
    encode_match_state_factors,
    encode_tactical_state_factors,
)
from ..env.projection import (
    project_action_space,
    project_observer_frame,
)
from ..types import ActionSpaceBinding


_SUMMARY_QUERY_FACTORS = None
_ZERO_NUMERIC_ROW = None


def _query_rows():
    global _SUMMARY_QUERY_FACTORS, _ZERO_NUMERIC_ROW
    if _SUMMARY_QUERY_FACTORS is None:
        import numpy as np
        _SUMMARY_QUERY_FACTORS = np.asarray([
            (Segment.MATCH_SUMMARY, TokenKind.QUERY, 2, 1, 0, 0, 0, 0, 0, 0),
            (Segment.KYOKU_SUMMARY, TokenKind.QUERY, 3, 1, 0, 0, 0, 0, 0, 0),
            (Segment.ACTOR_QUERY, TokenKind.QUERY, 1, 1, 0, 0, 0, 0, 0, 0),
        ], dtype=np.uint8)
        _ZERO_NUMERIC_ROW = np.zeros((1, 8), dtype=np.float32)
    return _SUMMARY_QUERY_FACTORS, _ZERO_NUMERIC_ROW


@dataclass(frozen=True, slots=True)
class PackedBatch:
    order: tuple[int, ...]
    batches: tuple[tuple[int, ...], ...]
    total_tokens: int
    padded_tokens: int


@dataclass(frozen=True, slots=True, eq=False)
class EncodedActionSpace:
    binding: ActionSpaceBinding
    token_factors: object
    token_numeric: object
    actor_query_index: int
    rank_boundary_features: object
    decision_seat: int
    action_factors: object
    action_representatives: tuple[int, ...]
    action_members: tuple[tuple[int, ...], ...]
    native_candidates: tuple[object, ...]

    def __eq__(self, other):
        if not isinstance(other, EncodedActionSpace):
            return NotImplemented
        import numpy as np
        return (
            self.binding == other.binding
            and np.array_equal(self.token_factors, other.token_factors)
            and np.array_equal(self.token_numeric, other.token_numeric)
            and self.actor_query_index == other.actor_query_index
            and np.array_equal(
                self.rank_boundary_features, other.rank_boundary_features
            )
            and self.decision_seat == other.decision_seat
            and np.array_equal(self.action_factors, other.action_factors)
            and self.action_representatives == other.action_representatives
            and self.action_members == other.action_members
            and self.native_candidates == other.native_candidates
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


def _start_boundary(frame, rows):
    """Restore scores/deposits fixed at the latest start-of-hand event."""
    result = dict(frame)
    start = next((row for row in reversed(rows) if int(row.get("kind", 0)) == 2), None)
    if start is None:
        return result
    payload = bytes(start.get("payload") or b"")
    if len(payload) >= 20 and payload[0] == 2:
        result["riichi_deposits"] = int.from_bytes(payload[2:4], "little")
        result["scores"] = tuple(
            int.from_bytes(payload[4 + 4 * seat:8 + 4 * seat], "little", signed=True)
            for seat in range(4)
        )
    args = start.get("args", (0, 0, 0, 0))
    result["honba"] = int(args[2])
    return result


def encode_native_batch(batch, histories, *, event_cache=None,
                        include_all_views=False,
                        include_public_snapshot=True):
    import numpy as np
    query_factors, zero_numeric_row = _query_rows()
    encoded = []
    for state in batch.transition.states:
        view_seats = range(4) if include_all_views else (
            int(space.seat) for space in state.action_spaces
        )
        for observer in view_seats:
            if include_all_views:
                frame, actor, space = project_observer_frame(
                    state, observer=int(observer),
                )
            else:
                space = next(
                    row for row in state.action_spaces if int(row.seat) == int(observer)
                )
            binding = ActionSpaceBinding(
                int(state.environment_id),
                int(state.episode_generation),
                (
                    int(state.frame_id)
                    if int(state.frame_id) > 0
                    else (1 << 63) | int(batch.transition.transition_id)
                ),
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
            if not include_all_views:
                frame, actor = project_action_space(
                    state,
                    space,
                    observer=observer,
                )
            frame["actor_concealed_counts"] = tuple(actor["concealed_counts"])
            match_factors, match_numeric = encode_match_state_factors(
                frame, observer=observer
            )
            tactical_factors, tactical_numeric = encode_tactical_state_factors(
                frame, actor, observer=observer,
                include_public_snapshot=include_public_snapshot,
            )
            actor_query = (
                len(match_factors) + 1 + len(event_factors)
                + len(tactical_factors) + 1
            )
            if space.candidates:
                actions = encode_native_actions(space.candidates, observer=observer)
            else:
                from .actions import EncodedActions
                actions = EncodedActions((), (), ())
            actor_query_factors = query_factors[2:3].copy()
            actor_query_factors[0, 8] = int(state.phase) & 0xFF
            actor_query_factors[0, 9] = int(bool(space.candidates))
            token_factors = np.concatenate((
                match_factors, query_factors[:1], event_factors,
                tactical_factors, query_factors[1:2], actor_query_factors,
            ), axis=0)
            token_numeric = np.concatenate((
                match_numeric, zero_numeric_row, event_numeric,
                tactical_numeric, zero_numeric_row, zero_numeric_row,
            ), axis=0)
            encoded.append(
                EncodedActionSpace(
                    binding,
                    token_factors,
                    token_numeric,
                    actor_query,
                    encode_rank_boundary(_start_boundary(frame, store.rows)),
                    (observer - int(frame["dealer"])) % 4,
                    np.asarray(actions.factors, dtype=np.uint8).reshape(-1, 15),
                    actions.representatives,
                    actions.members,
                    tuple(space.candidates),
                )
            )
    return tuple(encoded)


def model_batch(
    encoded, *, device="cpu", backend="sdpa", pin_memory=None,
    include_actions=True,
):
    import numpy as np
    import torch

    if not encoded:
        raise ValueError("cannot build an empty model batch")
    lengths_array = np.fromiter(
        (len(row.token_factors) for row in encoded), dtype=np.int64, count=len(encoded)
    )
    maximum = int(lengths_array.max())
    if pin_memory is None:
        pin_memory = torch.device(device).type == "cuda"
    pin_memory = bool(pin_memory and torch.cuda.is_available())

    def host_tensor(shape, dtype, *, zero=False):
        tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
        return tensor.zero_() if zero else tensor

    # Embedding accepts int32 indices on CPU and CUDA.  The encoded factors are
    # bytes, so widening them all the way to int64 only doubles transfer and
    # resident minibatch storage without increasing the representable domain.
    token_factors_host = host_tensor(
        (len(encoded), maximum, 10), torch.int32, zero=True
    )
    token_numeric_host = host_tensor(
        (len(encoded), maximum, 8), torch.float32, zero=True
    )
    token_factors_array = token_factors_host.numpy()
    token_numeric_array = token_numeric_host.numpy()
    action_lengths_array = action_factors_host = action_factors_array = None
    offsets = None
    if include_actions:
        action_lengths_array = np.fromiter(
            (len(row.action_factors) for row in encoded),
            dtype=np.int64,
            count=len(encoded),
        )
        action_maximum = int(action_lengths_array.max())
        action_factors_host = host_tensor(
            (len(encoded), action_maximum, 15), torch.int32, zero=True
        )
        action_factors_array = action_factors_host.numpy()
        offsets = [0]
    action_end = 0
    for index, row in enumerate(encoded):
        length = len(row.token_factors)
        token_factors_array[index, :length] = row.token_factors
        token_numeric_array[index, :length] = row.token_numeric
        if include_actions:
            next_action_end = action_end + len(row.action_factors)
            action_factors_array[index, :len(row.action_factors)] = row.action_factors
            action_end = next_action_end
            offsets.append(action_end)
    lengths_host = host_tensor((len(encoded),), torch.long)
    lengths_host.numpy()[:] = lengths_array
    actor_queries_host = host_tensor((len(encoded),), torch.long)
    actor_queries_host.numpy()[:] = np.fromiter(
        (row.actor_query_index for row in encoded), dtype=np.int64, count=len(encoded)
    )
    # Seat-relative score/round features are part of the public actor input,
    # not privileged critic data.  Materialize them for every model call.
    decision_seats_host = host_tensor((len(encoded),), torch.long)
    decision_seats_host.numpy()[:] = np.fromiter(
        (row.decision_seat for row in encoded),
        dtype=np.int64,
        count=len(encoded),
    )
    rank_boundary_host = host_tensor(
        (len(encoded), BOUNDARY_RANK_FEATURES), torch.float32
    )
    rank_boundary_host.numpy()[:] = np.stack([
        getattr(
            row, "rank_boundary_features",
            np.zeros(BOUNDARY_RANK_FEATURES, dtype=np.float32),
        )
        for row in encoded
    ])
    if include_actions:
        action_lengths_host = host_tensor((len(encoded),), torch.long)
        action_lengths_host.numpy()[:] = action_lengths_array
        action_offsets_host = host_tensor((len(offsets),), torch.long)
        action_offsets_host.numpy()[:] = offsets

    def transfer(tensor):
        return tensor.to(device=device, non_blocking=pin_memory)

    result = {
        "token_factors": transfer(token_factors_host),
        "token_numeric": transfer(token_numeric_host),
        "actor_query_indices": transfer(actor_queries_host),
        "lengths": transfer(lengths_host),
        "decision_seats": transfer(decision_seats_host),
        "rank_boundary_features": transfer(rank_boundary_host),
        "backend": backend,
    }
    if include_actions:
        result.update({
            "action_factors": transfer(action_factors_host),
            "action_lengths": transfer(action_lengths_host),
            "action_offsets": transfer(action_offsets_host),
        })
    return result


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
            batches.append(tuple(current))
            padded += current_max * len(current)
            current, current_max, current_sum = [], 0, 0
        current.append(index)
        current_max = max(current_max, lengths[index])
        current_sum += lengths[index]
    if current:
        batches.append(tuple(current))
        padded += current_max * len(current)
    return PackedBatch(tuple(order), tuple(batches), sum(lengths), padded)
