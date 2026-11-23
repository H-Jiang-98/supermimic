"""Pure checks for the fixed-input continuation evidence execution contract."""

from __future__ import annotations

REFERENCE_VERSION = "eit_live_update_reference_v3"
SUPPORTED_REFERENCE_VERSIONS = (
  "eit_live_update_reference_v1",
  "eit_live_update_reference_v2",
  REFERENCE_VERSION,
)

RESUME_EXECUTION_FIELDS = frozenset(
  {
    "torch_compile",
    "torchscript_optimized_execution",
    "torchscript_import_mode",
    "deterministic_algorithms",
    "fixed_input_assembly",
    "discarded_private_policy_warmup_updates",
    "cublas_workspace_config",
    "expected_optimizer_origin",
  }
)


def resume_execution_matches(receipt, live, process_environment=None):
  """Require the same deterministic, privately warmed execution on both sides."""
  if not isinstance(receipt, dict) or not isinstance(live, dict):
    return False
  execution = receipt.get("verification_execution")
  if (
    not isinstance(execution, dict)
    or not RESUME_EXECUTION_FIELDS.issubset(execution)
    or execution != live.get("verification_execution")
    or not isinstance(receipt.get("device"), str)
    or receipt["device"] != live.get("device")
  ):
    return False
  expected = {
    "torch_compile": "force_eager",
    "torchscript_optimized_execution": False,
    "deterministic_algorithms": True,
    "fixed_input_assembly": "cpu_contiguous_then_shared_device_constructors_v1",
    "discarded_private_policy_warmup_updates": 1,
    "expected_optimizer_origin": "deepcopy_of_live_parameter_optimizer_graph",
  }
  if any(
    type(execution.get(key)) is not type(value)
    or type(live["verification_execution"].get(key)) is not type(value)
    or execution[key] != value
    for key, value in expected.items()
  ):
    return False
  if execution.get("torchscript_import_mode") not in (
    "enabled",
    "disabled_before_import",
  ):
    return False
  if (
    receipt["device"].split(":")[0] == "cuda"
    and execution.get("cublas_workspace_config") != ":4096:8"
  ):
    return False
  if process_environment is not None:
    if not isinstance(process_environment, dict):
      return False
    import_mode = (
      "disabled_before_import"
      if process_environment.get("PYTORCH_JIT") == "0"
      else "enabled"
    )
    if execution["torchscript_import_mode"] != import_mode or execution.get(
      "cublas_workspace_config"
    ) != process_environment.get("CUBLAS_WORKSPACE_CONFIG"):
      return False
  return True
