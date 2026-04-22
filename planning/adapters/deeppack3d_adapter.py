# planning/adapters/deeppack3d_adapter.py
"""
DeepPack3DAdapter
- DP3D 설치형(import deeppack3d) 또는 경로형(DP3D_PATH, src) 모두 지원
- 입력(mm) -> DP3D 그리드 변환, 출력(그리드) -> mm 변환 일관 적용
- 현재 bin의 고정 아이템을 점유(occupied)로 추출해 DP3D 후보를 필터링(누적 유지)
- 플랜 캐시(self._plan)와 커서(self._cursor)로 매 stack() 호출 시 1개(or 소수)씩 소비
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import os, sys
import importlib
import importlib.util
from typing import List, Any


from planning.item import RotationType

# =========================
# 상수 / 단위 변환 유틸
# =========================
GRID_PER_MM = 32.0 / 1000.0
MM_PER_GRID = 1000.0 / 32.0

# =========================
# 어댑터 본체
# =========================
class DeepPack3DAdapter:
    def __init__(self, **kwargs):
        """
        args:
          method: "rl" | "bl" | "baf" | "bssf" | "blsf" (등)
          lookahead: int
          verbose: 0/1
        """
        args = kwargs.get("args", {}) or kwargs
        self.args = type("Args", (), {})()
        setattr(self.args, "method",  args.get("method", "rl"))
        setattr(self.args, "verbose", int(args.get("verbose", 0)))
        setattr(self.args, "lookahead", int(args.get("lookahead", 3)))  # ✅ 기본값

        self._dp3d = None
        
        self.item_list_cache: List[Any] = []  # dp3d가 입력으로 사용할 아이템 리스트 캐시
    # -------------------------
    # DP3D 모듈 로딩
    # -------------------------
    def _lazy_import(self) -> None:
        if self._dp3d is not None:
            return

        # 1) 설치형 우선 시도
        try:
            from deeppack3d import deeppack3d as _fn  # type: ignore
            self._dp3d = _fn
            return
        except ModuleNotFoundError:
            pass
        except Exception as e:
            raise RuntimeError(f"'deeppack3d' import 실패: {e}")

        # 2) 경로형 (DP3D_PATH 및 src 주입 후 정식 import)
        dp3d_root = os.environ.get("DP3D_PATH", "").strip()
        if not dp3d_root:
            raise ModuleNotFoundError(
                "deeppack3d가 설치되어 있지 않고 DP3D_PATH도 비어있습니다.\n"
                "  - pip install -e . 로 설치하거나\n"
                "  - export DP3D_PATH=/home/USER/study/DeepPack3D"
            )
        root = Path(dp3d_root).expanduser().resolve()
        if not root.exists():
            raise ModuleNotFoundError(f"DP3D_PATH가 존재하지 않습니다: {root}")

        # sys.path 주입
        for p in (str(root), str(root / "src")):
            if p and p not in sys.path and Path(p).exists():
                sys.path.insert(0, p)

        # 정식 import
        last_err = None
        try:
            mod = importlib.import_module("deeppack3d")
            if hasattr(mod, "deeppack3d"):
                self._dp3d = getattr(mod, "deeppack3d")
                return
        except Exception as e:
            last_err = e

        # 파일 직접 로딩 폴백
        candidates = [
            root / "deeppack3d.py",
            root / "deeppack3d" / "__init__.py",
            root / "src" / "deeppack3d.py",
            root / "src" / "deeppack3d" / "__init__.py",
        ]
        for p in candidates:
            if p.is_file():
                spec = importlib.util.spec_from_file_location("deeppack3d", str(p))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["deeppack3d"] = mod
                    spec.loader.exec_module(mod)  # type: ignore
                    if hasattr(mod, "deeppack3d"):
                        self._dp3d = getattr(mod, "deeppack3d")
                        return

        hint = f"sys.path 주입: {[str(root), str(root/'src')]} | 마지막 오류: {last_err}"
        raise ModuleNotFoundError(
            "DP3D 로딩 실패. 다음 구조를 확인하세요:\n"
            "  - <DP3D_PATH>/deeppack3d.py\n"
            "  - <DP3D_PATH>/deeppack3d/__init__.py\n"
            "  - <DP3D_PATH>/src/deeppack3d.py\n"
            "  - <DP3D_PATH>/src/deeppack3d/__init__.py\n" + hint
        )

    # -------------------------
    # DP3D 호출 시그니처 폴백
    # -------------------------
    def _run_dp3d(self, path: Path, **overrides):
        """
        - 기본은 파일 입력(data='file', path=...)
        - self.args.* 또는 overrides로 n_iterations/seed/train/visualize/batch_size/verbose 지정 가능
        """
        fn = self._dp3d

        # 1) 공통 파라미터 구성 (self.args → overrides 순으로 병합)
        def _get(name, default):
            return overrides.get(name, getattr(self.args, name, default))

        call_kwargs = dict(
            n_iterations = int(_get("n_iterations", 100)),
            lookahead     = int(_get("lookahead", 3)),
            seed         = _get("seed", None),
            verbose      = int(_get("verbose", 1)),
            data         = _get("data", "file"),
            path         = str(_get("path", str(path))),
            train        = bool(_get("train", False)),
            visualize    = bool(_get("visualize", False)),
            batch_size   = int(_get("batch_size", 32)),
        )

        # 2) 최신(named) 시그니처 시도
        try:
            print(f"DP3D 호출: method={self.args.method}, kwargs={call_kwargs}")
            return fn(self.args.method, **call_kwargs)
        except TypeError:
            pass

        # 3) 구버전(위치 인자형) 폴백
        try:
            return fn(
                self.args.method,
                call_kwargs['lookahead'],
                int(call_kwargs["n_iterations"]),
                call_kwargs["data"],
                str(call_kwargs["path"]),
                int(call_kwargs["verbose"]),
            )
        except TypeError:
            pass

        # 4) 가장 축약된 폴백(아주 오래된 버전)
        return fn(self.args.method, str(path))


    def stack(self, bin_obj, items_list):
        """
        DP3D 제너레이터를 끝까지 받아 proposals를 만들고,
        lookahead 창(= self.args.lookahead)으로 슬라이딩하며
        창 안에서 checkPivot_R 기준으로 실제 적재 가능한 항목이 있으면
        즉시 그 아이템을 store()하고 같은 창을 다시 검사한다.
        창에서 더 이상 적재할 수 없으면 창을 한 칸 앞으로 밀고 반복한다.
        결과는 fit_result[i] = 1(적재) / 0(미적재).
        """
        # ── 준비 ─────────────────────────────────────────────────────
        n = len(items_list)
        fit_result = [-1] * n
        if bin_obj is None or n == 0:
            return fit_result

        # 안전한 로컬 임포트 (경로 환경에 따라)
        try:
            from utils.checkPivot import checkPivot_R
        except Exception:
            from planning.utils.checkPivot import checkPivot_R  # type: ignore

        def _to_grid(v_mm: float) -> int:
            return int(round(float(v_mm) * GRID_PER_MM))

        def _to_mm_dims_from_grid(size_grid):
            gw, gh, gd = map(int, size_grid)
            return (gw * MM_PER_GRID, gh * MM_PER_GRID, gd * MM_PER_GRID)

        def _to_mm_xyz_from_grid(pos_grid):
            gx, gy, gz = map(int, pos_grid)
            return [gx * MM_PER_GRID, gy * MM_PER_GRID, gz * MM_PER_GRID]

        def _dims_close_mm(a_mm, b_mm, tol=MM_PER_GRID / 2.0):
            return (abs(a_mm[0] - b_mm[0]) <= tol and
                    abs(a_mm[1] - b_mm[1]) <= tol and
                    abs(a_mm[2] - b_mm[2]) <= tol)

        # README 형태: (_, (x,y,z), (w,h,d), _)
        def _parse_dp3d_tuple(res):
            pos_g = None; size_g = None
            if isinstance(res, tuple):
                if len(res) >= 3:
                    if isinstance(res[1], (list, tuple)) and len(res[1]) >= 3:
                        pos_g = tuple(map(int, res[1][:3]))
                    if isinstance(res[2], (list, tuple)) and len(res[2]) >= 3:
                        size_g = tuple(map(int, res[2][:3]))
            elif isinstance(res, dict):
                if all(k in res for k in ('x','y','z')):
                    pos_g = (int(res['x']), int(res['y']), int(res['z']))
                elif 'pos' in res and isinstance(res['pos'], (list, tuple)) and len(res['pos']) >= 3:
                    pos_g = tuple(map(int, res['pos'][:3]))
                if all(k in res for k in ('w','h','d')):
                    size_g = (int(res['w']), int(res['h']), int(res['d']))
                elif 'size' in res and isinstance(res['size'], (list, tuple)) and len(res['size']) >= 3:
                    size_g = tuple(map(int, res['size'][:3]))
            return pos_g, size_g

        def _write_items_file(items, path: Path):
            with path.open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(f"{_to_grid(it.width)} {_to_grid(it.height)} {_to_grid(it.depth)}\n")

        # ── 1) DP3D 제너레이터를 끝까지 받아 proposals 생성 ─────────
        self._lazy_import()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path  = Path(tmpdir)
            items_txt = tmp_path / "items.txt"
            _write_items_file(items_list, items_txt)

            dp3d_gen = self._run_dp3d(
                items_txt,  # path 인자는 위치 인자로만 한 번 전달
                n_iterations=-1,
                lookahead=getattr(self.args, "lookahead", 3),  # ✅ 오타 수정 + 기본값
                data='file',
            )
            proposals = []  # [(pos_mm, size_mm)]
            for res in dp3d_gen:
                if res is None:  # 새 bin 신호 → 단일 bin 기준에선 스킵
                    continue
                pos_g, size_g = _parse_dp3d_tuple(res)
                if size_g is None:
                    continue
                proposals.append((
                    _to_mm_xyz_from_grid(pos_g) if pos_g else [0, 0, 0],
                    _to_mm_dims_from_grid(size_g),
                ))

        # ── 2) 슬라이딩 윈도우(lookahead)로 그리디 적재 ─────────────
        L = max(1, int(getattr(self.args, "lookahead", 3)))
        used = [False] * n  # 같은 인덱스 재사용 방지

        # size_mm → (idx, rot) 매칭 헬퍼 (WHD 우선, 실패 시 HWD)
        def _match_item(size_mm):
            # WHD 먼저
            for i, it in enumerate(items_list):
                if used[i]: continue
                whd = (float(it.width), float(it.height), float(it.depth))
                if _dims_close_mm(whd, size_mm):
                    return i, RotationType.RT_WHD
            # HWD
            for i, it in enumerate(items_list):
                if used[i]: continue
                hwd = (float(it.height), float(it.width), float(it.depth))
                if _dims_close_mm(hwd, size_mm):
                    return i, RotationType.RT_HWD
            return None, None

        p = 0
        while p < len(proposals):
            window = proposals[p:p+L]
            placed_in_this_window = False

            # 창 안에서 하나라도 실제 적재 가능하면 즉시 커밋하고 같은 창 다시 검사
            for pos_mm, size_mm in window:
                idx, rot = _match_item(size_mm)
                if idx is None:
                    continue

                cand = items_list[idx]
                cand.b_position = pos_mm
                cand.rotation_quat = rot

                fit, _ = checkPivot_R(
                    bin_obj,
                    cand,
                    pivot_pos=cand.b_position,
                    rotation_quat=cand.rotation_quat,
                    apply_margin=True,
                )
                # fit > 0 이면 실제로 store 수행
                if fit > 0:
                    bin_obj.store(cand)
                    used[idx] = True
                    fit_result[idx] = 1
                    placed_in_this_window = True
                    break  # 같은 창을 다시 검사(연쇄 배치 시도)

            if placed_in_this_window:
                # 같은 시작 p에서 창을 다시 검사 (방금 배치로 인해 창 내 다른 제안이 현실화될 수 있음)
                continue
            else:
                # 창 내에 적재 가능한 것이 없으면 창을 한 칸 앞으로
                p += 1

        # 못 넣은 것들 0으로
        for i in range(n):
            if fit_result[i] == -1:
                fit_result[i] = 0

        return fit_result


