import hashlib

import riichi


def digest(threads):
    env = riichi.Env(32, master_seed=0x5EED, num_threads=threads, privileged=True)
    transition = env.reset(list(range(32)))
    events = []
    for _ in range(12):
        events.extend((event.environment_id, event.sequence, event.kind) for event in transition.events)
        actions = [d.actions[0] for state in transition.states for d in state.decisions]
        transition = env.step(actions)
    payload = b"".join(bytes(value) for value in env.snapshot(list(range(32))).values())
    return hashlib.sha256(payload).hexdigest(), tuple(events)


def test_thread_count_does_not_change_state_or_event_digest():
    assert digest(1) == digest(4)


def test_inspect_emits_no_events_and_restore_continues_exactly():
    env = riichi.Env(2, master_seed=9, num_threads=2)
    reset = env.reset([0, 1])
    assert reset.events
    inspected = env.inspect([0, 1])
    assert not inspected.events
    snapshots = env.snapshot([0, 1])
    actions = [d.actions[0] for state in inspected.states for d in state.decisions]
    expected = env.step(actions)
    expected_bytes = env.snapshot([0, 1])
    restored = env.restore(snapshots)
    assert not restored.events
    replay_actions = [d.actions[0] for state in restored.states for d in state.decisions]
    actual = env.step(replay_actions)
    assert [e.kind for e in actual.events] == [e.kind for e in expected.events]
    assert env.snapshot([0, 1]) == expected_bytes
