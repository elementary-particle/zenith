from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util
from zenith_ppo.metrics import TensorBoardProjector


def test_scalar_projection_uses_canonical_tag_step_value(tmp_path):
    projector = TensorBoardProjector(tmp_path, flush_seconds=1)
    projector.enqueue([{"name": "ppo/policy_loss", "value": .1, "step": 3}]); projector.close()
    events = EventAccumulator(str(tmp_path)); events.Reload()
    scalar = events.Tensors("ppo/policy_loss")[0]
    value = float(tensor_util.make_ndarray(scalar.tensor_proto))
    assert scalar.step == 3 and abs(value - .1) <= 1e-12
