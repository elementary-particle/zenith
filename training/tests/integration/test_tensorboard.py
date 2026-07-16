from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from zenith_ppo.metrics import TensorBoardProjector


def test_scalar_projection_uses_canonical_tag_step_value(tmp_path):
    projector = TensorBoardProjector(tmp_path, flush_seconds=1)
    projector.enqueue([{"name": "ppo/policy_loss", "value": .1, "step": 3}]); projector.close()
    events = EventAccumulator(str(tmp_path)); events.Reload()
    scalar = events.Tensors("ppo/policy_loss")[0]
    value = float(tensor_util.make_ndarray(scalar.tensor_proto))
    assert scalar.step == 3 and abs(value - .1) <= 1e-12


def test_rollout_progress_projects_bot_relative_win_rate(tmp_path):
    projector = TensorBoardProjector(tmp_path, run_id="run", writer_session="one", flush_seconds=1)
    projector.enqueue(({"name": "rollout_rating/win_rate_vs_conservative_bot",
                        "value": .55, "step": 3},))
    projector.close()

    event_files = list(tmp_path.rglob("events.out.tfevents.*"))
    assert len(event_files) == 1
    events = EventAccumulator(str(tmp_path)); events.Reload()
    assert "rollout_rating/win_rate_vs_conservative_bot" in events.Tags()["tensors"]


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
