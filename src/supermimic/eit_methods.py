"""Method names shared by EIT configuration, launch and checkpoint contracts."""

from typing import Literal

LOCAL_METHODS = ("prop_kp", "gt", "mean", "belief")
BASELINES = ("prop_kp", "gt")
EstimatorGroup = Literal["A", "B", "C", "prop_kp", "gt", "mean", "belief"]
