# planning/_stubs.py
#
# Standalone stubs for hardware-specific modules that exist in the full
# PalletFit_RL repo but are NOT needed for the planning/simulation code
# released as part of this paper.
#
# - LogicalState   : base class originally from motion.network_element
# - _RobotStub     : replaces dsrpy.Robot (Doosan robot SDK)
# - N_detection    : replaces detection.N_detection (camera perception)

from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# LogicalState  (from motion/network_element.py)
# ---------------------------------------------------------------------------
LogicalStateMonitorType = Callable[["LogicalState"], None]


class LogicalState:
    """Minimal base class for Packer.  Robot-specific methods are no-ops."""

    def __init__(self):
        self.monitors: list = []

    def add_monitor(self, monitor: LogicalStateMonitorType) -> None:
        self.monitors.append(monitor)

    def add_monitors(self, monitors: Sequence[LogicalStateMonitorType]) -> None:
        self.monitors.extend(monitors)

    def reset(self):
        pass

    # ---- robot / network stubs (originally in LogicalState subclasses) ----
    def sync_targetBin_state(self):
        pass

    def update_robot_state(self, **kwargs):
        pass

    @property
    def last_posori_state(self):
        return [0.0] * 7

    @property
    def joint_posori_dict(self):
        return {"doosan_home": {"joints": [0.0] * 6}}


# ---------------------------------------------------------------------------
# Robot stub  (replaces dsrpy.Robot)
# ---------------------------------------------------------------------------
class _RobotStub:
    """No-op stub for the Doosan robot SDK (dsrpy).
    All methods silently do nothing so planning code runs without hardware."""

    def open_connection(self, ip: str) -> None:
        pass

    def connect_rt(self, ip: str) -> None:
        pass

    def servo_on(self) -> None:
        pass

    def servo_off(self) -> None:
        pass

    def disconnect_rt(self) -> None:
        pass

    def close_connection(self) -> None:
        pass

    def set_tcp_tmat(self, tmat) -> None:
        pass


class _DsrModule:
    """Minimal stand-in for the `dsrpy` package."""
    Robot = _RobotStub


# expose as module-level alias so `import dsrpy as dsr` can be replaced by
#   from planning._stubs import dsr
dsr = _DsrModule()


# ---------------------------------------------------------------------------
# N_detection stub  (replaces detection.N_detection)
# ---------------------------------------------------------------------------
def N_detection():
    """Returns None — no camera available in simulation/planning mode."""
    return None
