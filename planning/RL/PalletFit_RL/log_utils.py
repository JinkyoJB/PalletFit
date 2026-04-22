# planning/RL/PalletFit_RL/log_utils.py
from __future__ import annotations
import os, logging, datetime
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter  # PyTorch
except Exception:
    SummaryWriter = None  # 텐서보드 미사용 fallback

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_log_dir(root: str = "planning/RL/PalletFit_RL/logs", ts: Optional[str] = None) -> str:
    ts = ts or _timestamp()
    log_dir = os.path.join(root, ts)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

class LogManager:
    """
    - TensorBoard SummaryWriter: log_dir=.../logs/{시작시간}
    - Python logging: 콘솔 + 파일(info.txt) 동시 출력
    """
    def __init__(self, log_dir: str, use_tb: bool = True, logger_name: str = "PalletFit"):
        self.log_dir = log_dir
        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir) if (use_tb and SummaryWriter is not None) else None

        # Python logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 중복 출력 방지

        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

        # 기존 핸들러 제거(재초기화 시 중복 방지)
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)

        # 콘솔
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # 파일
        fh = logging.FileHandler(os.path.join(log_dir, "info.txt"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

    # ===== TB helpers =====
    def scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def scalars(self, tag: str, metrics: Dict[str, float], step: int):
        if self.writer is not None:
            # add_scalars는 여러 곡선 묶음에 유용
            self.writer.add_scalars(tag, {k: float(v) for k, v in metrics.items()}, step)

    # ===== Text log =====
    def info(self, msg: str):
        self.logger.info(msg)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        # 파일핸들 닫기
        for h in list(self.logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            self.logger.removeHandler(h)
