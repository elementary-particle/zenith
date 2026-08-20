import hashlib
import numpy as np
import riichi


def digest(workers):
    env = riichi.Env(4, master_seed=9, num_threads=workers)
    transition = env.reset([0, 1, 2, 3])
    for _ in range(16):
        actions = tuple(
            space.candidates[0].select()
            for state in transition.states
            for space in state.action_spaces
        )
        if actions:
            transition = env.step(actions)
        else:
            transition = env.advance([
                int(state.environment_id)
                for state in transition.states
                if int(state.lifecycle) not in (3, 4)
            ])
    values = transition.as_numpy()
    hands = np.stack([
        np.frombuffer(bytes(decision.concealed_counts), dtype=np.uint8)
        for state in transition.states
        for decision in state.action_spaces
    ])
    efficiency = riichi.evaluate_hand_efficiency(
        hands, np.zeros(len(hands), dtype=np.uint8)
    )
    result = hashlib.sha256(
        values["state_scores"].tobytes()
        + values["event_kind"].tobytes()
        + values["event_args"].tobytes()
        + efficiency.shanten.tobytes()
        + efficiency.improving_tile_mask.tobytes()
    ).hexdigest()
    env.close()
    return result


def test_fixed_seed_state_event_digest_ignores_thread_count():
    assert len({digest(workers) for workers in (1, 2, 4, 8)}) == 1
