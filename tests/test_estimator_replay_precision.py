import pytest
import torch

from supermimic.scripts.check_eit_replay_precision import factor_round_trip
from supermimic.utils.validation import check_covariance


@pytest.mark.parametrize(
  ("dtype", "off_diagonal"),
  [(torch.float16, 0.08853765577), (torch.bfloat16, 0.09058165550)],
)
def test_raw_half_can_break_psd_and_factor_codec_retains_singular_frame(
  dtype, off_diagonal
):
  vector = torch.zeros(70)
  vector[:2] = torch.tensor([1, off_diagonal])
  singular = vector[:, None] * vector[None, :]
  tolerance = 128 * torch.finfo(torch.float32).eps
  check_covariance(singular, "reference", rtol=tolerance)
  with pytest.raises(ValueError, match="positive semidefinite"):
    check_covariance(singular.to(dtype).float(), "rounded", rtol=tolerance)

  # The positive definite frame exercises compressed storage in the same batch.
  positive = singular + torch.eye(70) * 1e-4
  covariance = torch.stack([singular, positive])
  restored, payload, fallbacks = factor_round_trip(covariance, dtype)
  assert fallbacks == 1
  assert payload == (2 + 2485 * (4 + 2) + 70 * 4)
  assert torch.equal(restored[0], singular)
  check_covariance(restored, "restored", rtol=tolerance)
