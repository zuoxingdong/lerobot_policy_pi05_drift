"""Pi0.5-Drift — a LeRobot out-of-tree plugin policy (Path A).

A faithful vendored copy of LeRobot's Pi0.5 (upstream lerobot 0.6.0) plus the one-step
"Drifting" (DBPO) objective, 1-NFE inference, and optional KeyStone test-time selection.
"""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package "
        "(e.g. `pip install 'lerobot[pi,dataset]>=0.6.0,<0.7'`)."
    ) from exc

from .configuration_pi05_drift import PI05DriftConfig
from .modeling_pi05_drift import PI05DriftPolicy
from .processor_pi05_drift import make_pi05_drift_pre_post_processors

__all__ = [
    "PI05DriftConfig",
    "PI05DriftPolicy",
    "make_pi05_drift_pre_post_processors",
]
