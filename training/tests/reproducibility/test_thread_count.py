import hashlib
import numpy as np
import riichi


def digest(workers):
    env = riichi.Env(4, master_seed=9, num_threads=workers)
    transition = env.reset([0, 1, 2, 3])
    for _ in range(16):
        transition = env.step(tuple(
            decision.actions[0]
            for state in transition.states
            for decision in state.decisions
        ))
    values = transition.as_numpy()
    hands = np.stack([
        np.frombuffer(bytes(decision.concealed_counts), dtype=np.uint8)
        for state in transition.states
        for decision in state.decisions
    ])
    analysis = riichi.analyze_hands(hands, np.zeros(len(hands), dtype=np.uint8))
    result = hashlib.sha256(
        values["state_scores"].tobytes()
        + values["event_kind"].tobytes()
        + values["event_args"].tobytes()
        + analysis.shanten.tobytes()
        + analysis.improving_type_mask.tobytes()
    ).hexdigest()
    env.close()
    return result


def test_fixed_seed_state_event_digest_ignores_thread_count():
    assert len({digest(workers) for workers in (1, 2, 4, 8)}) == 1
