import gzip
import json
import zipfile

from zenith_ppo.bc.data import (
    ArchiveCorpus,
    HAND_OUTCOME_DEAL_IN,
    HAND_OUTCOME_DRAW,
    HAND_OUTCOME_OTHER_WIN,
    HAND_OUTCOME_WIN,
    _event_group,
    _hand_outcome_targets,
    physicalize_kyoku,
    replay_game,
    replay_games,
)
from zenith_ppo.cli.train_bc import _load_chunks


def _deck():
    tiles = []
    for suit in "mps":
        for rank in range(1, 10):
            tiles.extend(
                [f"{rank}{suit}r", f"{rank}{suit}", f"{rank}{suit}", f"{rank}{suit}"]
                if rank == 5 else [f"{rank}{suit}"] * 4
            )
    for honor in "ESWNPFC":
        tiles.extend([honor] * 4)
    return tiles


def _take(deck, tile=None):
    index = 0 if tile is None else deck.index(tile)
    return deck.pop(index)


def _fixture(*, post_kan_dora=False):
    deck = _deck()
    dora = _take(deck, "1m")
    second_dora = _take(deck, "2p") if post_kan_dora else None
    hands = [[] for _ in range(4)]
    if post_kan_dora:
        hands[0].extend(_take(deck, "2m") for _ in range(4))
    for hand in hands:
        while len(hand) < 13:
            hand.append(_take(deck))
    first_draw = _take(deck)
    rows = [{
        "type": "start_kyoku", "bakaze": "E", "dora_marker": dora,
        "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000] * 4, "tehais": hands,
    }, {"type": "tsumo", "actor": 0, "pai": first_draw}]
    if post_kan_dora:
        replacement = _take(deck)
        rows.extend([
            {"type": "ankan", "actor": 0, "consumed": ["2m"] * 4},
            {"type": "tsumo", "actor": 0, "pai": replacement},
            {"type": "dora", "dora_marker": second_dora},
        ])
    else:
        rows.append({
            "type": "dahai", "actor": 0, "pai": first_draw, "tsumogiri": True,
        })
    rows.extend([
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
    ])
    return rows


def test_physical_wall_conserves_copies_red_fives_and_draw_positions():
    kyoku = physicalize_kyoku(_fixture())
    wall = kyoku.hanchan["wall"]

    assert sorted(wall) == list(range(136))
    assert {16, 52, 88} <= set(wall)
    assert wall[52] == kyoku.events[0]["tile"]
    assert wall[130] // 4 == 0
    assert kyoku.events[1]["tile"] == kyoku.events[0]["tile"]
    assert kyoku.events[1]["tsumogiri"] is True


def test_post_kan_draw_and_dora_retain_source_observation_order():
    kyoku = physicalize_kyoku(_fixture(post_kan_dora=True))
    assert [event["kind"] for event in kyoku.events[:4]] == [
        "tsumo", "ankan", "tsumo", "dora",
    ]
    assert kyoku.hanchan["wall"][135] == kyoku.events[2]["tile"]
    assert kyoku.hanchan["wall"][128] == kyoku.events[3]["dora_marker"]


def test_kan_input_stops_before_automatic_draw_and_dora():
    events = (
        {"kind": "kakan"}, {"kind": "tsumo"},
        {"kind": "dora"}, {"kind": "dahai"},
    )
    assert _event_group(events, 0, 1) == events[:1]


def test_consecutive_delayed_dora_group_includes_the_real_action():
    events = (
        {"kind": "dora"}, {"kind": "ankan"}, {"kind": "dora"},
    )
    assert _event_group(events, 0, 1) == events[:2]


def test_archive_reader_accepts_plain_and_gzip_members(tmp_path):
    rows = [
        {"type": "start_game", "aka_flag": True, "kyoku_first": 0},
        *_fixture(),
    ]
    payload = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("plain.mjson", payload)
        output.writestr("compressed.mjson", gzip.compress(payload))

    corpus = ArchiveCorpus([archive])
    assert len(corpus.members) == 2
    assert corpus.read((archive, "plain.mjson")) == tuple(rows)
    assert corpus.read((archive, "compressed.mjson")) == tuple(rows)


def test_whole_game_rejections_keep_typed_reason_counts(tmp_path):
    archive = tmp_path / "malformed.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "bad.mjson",
            json.dumps({"type": "start_game", "aka_flag": False, "kyoku_first": 0}),
        )
    corpus = ArchiveCorpus([archive])
    rows, accepted, rejected = corpus.collect(corpus.members, maximum=10)
    assert rows == ()
    assert accepted == ()
    assert rejected == {"ValueError:unsupported_game_profile": 1}


def test_batched_replay_matches_independent_replay_targets():
    rows = _fixture()
    expected = replay_game(rows)

    actual = replay_games((rows, rows), num_threads=2)

    def signature(examples):
        return tuple(
            (row.target, row.family, len(row.encoded.token_factors))
            for row in examples
        )
    assert tuple(map(signature, actual)) == (signature(expected),) * 2


def test_replay_marks_one_final_order_target_at_each_kyoku_boundary():
    examples = replay_game(_fixture())
    supervised = [
        row for row in examples if row.critic.rank_boundary_supervision
    ]

    assert len(supervised) == 1
    assert 0 <= supervised[0].critic.rank_order_target < 24
    assert all(
        row.critic.rank_order_target == supervised[0].critic.rank_order_target
        for row in examples
    )


def test_replay_attaches_training_only_hand_outcomes_and_score_deltas():
    examples = replay_game(_fixture())

    assert examples
    assert all(
        row.critic.hand_outcome_target == HAND_OUTCOME_DRAW
        for row in examples
    )
    assert all(row.critic.hand_score_delta == 0 for row in examples)


def test_multi_ron_outcomes_distinguish_winners_discarder_and_bystander():
    events = (
        {"kind": "hora", "actor_seat": 1, "target_seat": 0},
        {"kind": "hora", "actor_seat": 2, "target_seat": 0},
    )

    assert _hand_outcome_targets(events) == (
        HAND_OUTCOME_DEAL_IN,
        HAND_OUTCOME_WIN,
        HAND_OUTCOME_WIN,
        HAND_OUTCOME_OTHER_WIN,
    )


def test_original_member_is_streamed_each_time_without_native_objects(
    tmp_path, monkeypatch
):
    rows = [
        {"type": "start_game", "aka_flag": True, "kyoku_first": 0},
        *_fixture(),
    ]
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "game.mjson",
            b"".join(json.dumps(row).encode() + b"\n" for row in rows),
        )
    corpus = ArchiveCorpus([archive])
    member = corpus.members[0]
    native_read = corpus.read
    reads = 0

    def counted_read(value):
        nonlocal reads
        reads += 1
        return native_read(value)

    monkeypatch.setattr(corpus, "read", counted_read)
    first = corpus.examples(member)
    second = corpus.examples(member)

    assert reads == 2
    assert [(row.target, row.family) for row in second] == [
        (row.target, row.family) for row in first
    ]
    assert all(not row.encoded.native_candidates for row in second)
    assert all(not row.encoded.action_members for row in second)


def test_spawned_prefetch_worker_streams_original_archive(tmp_path):
    rows = [
        {"type": "start_game", "aka_flag": True, "kyoku_first": 0},
        *_fixture(),
    ]
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "game.mjson",
            b"".join(json.dumps(row).encode() + b"\n" for row in rows),
        )
    corpus = ArchiveCorpus([archive])
    chunks = tuple(_load_chunks(
        corpus,
        ((str(archive), "game.mjson", None),),
        workers=1,
        replay_batch=1,
        replay_threads=1,
    ))

    assert len(chunks) == 1
    assert chunks[0][0][0] is True
    assert len(chunks[0][0][1]) == 2
    assert not tuple(tmp_path.rglob("*.pkl"))
