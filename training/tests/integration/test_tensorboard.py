from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
import pytest

from zenith_ppo.metrics import TensorBoardProjector, tensorboard_records


def test_scalar_projection_uses_canonical_tag_step_value(tmp_path):
    projector = TensorBoardProjector(tmp_path, flush_seconds=1)
    projector.enqueue([{"name": "ppo/policy_loss", "value": .1, "step": 3}])
    projector.close()
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    scalar = events.Tensors("ppo/policy_loss")[0]
    value = float(tensor_util.make_ndarray(scalar.tensor_proto))
    assert scalar.step == 3 and abs(value - .1) <= 1e-12


def test_game_outcome_scalars_are_projected(tmp_path):
    values = {
        "game/player_average_winning_points": 7_700.0,
        "game/player_average_deal_in_points": 8_200.0,
        "game/exhaustive_ryukyoku_rate": 0.12,
        "game/player_bankrupt_rate": 0.01,
        "game/player_average_turns_before_winning": 8.4,
    }
    projector = TensorBoardProjector(tmp_path, flush_seconds=1)
    projector.enqueue(tensorboard_records(tuple(
        {"name": name, "value": value, "step": 256}
        for name, value in values.items()
    )))
    projector.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert set(values) <= set(events.Tags()["tensors"])
    for name, expected in values.items():
        scalar = events.Tensors(name)[0]
        assert scalar.step == 256
        assert float(tensor_util.make_ndarray(scalar.tensor_proto)) == pytest.approx(
            expected
        )


def test_rollout_progress_projects_bot_relative_win_rate(tmp_path):
    projector = TensorBoardProjector(tmp_path, run_id="run", writer_session="one", flush_seconds=1)
    projector.enqueue(({"name": "rollout_rating/rolling_win_rate_vs_conservative_bot",
                        "value": .55, "step": 3},))
    projector.close()

    event_files = list(tmp_path.rglob("events.out.tfevents.*"))
    assert len(event_files) == 1
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert "rollout_rating/rolling_win_rate_vs_conservative_bot" in events.Tags()["tensors"]


def test_resume_reuses_logical_run_and_purges_abandoned_tail(tmp_path):
    first = TensorBoardProjector(tmp_path, run_id="run", writer_session="one", flush_seconds=1)
    first.enqueue(({"name": "ppo/policy_loss", "value": 1.0, "step": 1},
                   {"name": "ppo/policy_loss", "value": 99.0, "step": 3}))
    first.close()
    resumed = TensorBoardProjector(
        tmp_path, run_id="run", writer_session="two", flush_seconds=1, purge_step=3
    )
    resumed.enqueue(({"name": "ppo/policy_loss", "value": 2.0, "step": 3},))
    resumed.close()
    event_files = sorted(tmp_path.rglob("events.out.tfevents.*"))
    assert len(event_files) == 2
    resume_events = tuple(EventFileLoader(str(event_files[-1])).Load())
    assert any(
        event.step == 3
        and event.HasField("session_log")
        and event.session_log.status == event_pb2.SessionLog.START
        for event in resume_events
    )
