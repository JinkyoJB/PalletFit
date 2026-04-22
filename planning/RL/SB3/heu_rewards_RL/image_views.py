# planning/RL/SB3/heu_rewards_RL/image_views.py
from __future__ import annotations
from typing import Literal, Tuple, Optional
import numpy as np
from PIL import Image

class MultiViewRenderer:
    """
    3-뷰(HWC uint8):
      - top   : (width, height)
      - front : (width, depth)
      - left  : (height, depth)

    use_fast2d=True  → 증분 버퍼로 세 뷰 모두 관리
    use_fast2d=False → 엔진 렌더(top/가능하면 front/left), 실패 시 간단 투영으로 대체
    """
    def __init__(
        self,
        *,
        img_h: int,
        img_w: int,
        use_fast2d: bool = False,
        fast2d_mode: Literal["count", "alpha"] = "alpha",
    ):
        self.H = int(img_h)
        self.W = int(img_w)
        self.use_fast2d = bool(use_fast2d)
        self.fast2d_mode = fast2d_mode

        self.bin = None

        # 월드→픽셀 스케일 (가로/세로 축 순서 주의)
        self._sx_top = self._sy_top = 1.0      # (width,  height)
        self._sx_front = self._sy_front = 1.0  # (width,  depth)
        self._sx_left = self._sy_left = 1.0    # (height, depth)

        # 증분 버퍼 occ: 겹침 카운트 / shade: 그레이스케일, 명암도 /  img: RGB, 실제관측에 넣는 용도
        self._occ_top   = self._shade_top   = self._img_top   = None
        self._occ_front = self._shade_front = self._img_front = None
        self._occ_left  = self._shade_left  = self._img_left  = None


    # ---------- utils ----------
    @staticmethod
    def hwc_to_chw_f32(x: np.ndarray) -> np.ndarray:
        '''
        축 순서를 (H, W, C) → (C, H, W)로 바꿔 PyTorch 관례(CHW)**로 변환
        '''
        a = x.astype(np.float32, copy=False)
        if a.dtype != np.float32 or a.max() > 1.0:
            a = a / 255.0
        return np.transpose(a, (2, 0, 1))

    # ---------- public API ----------
    def attach_bin(self, bin_obj) -> None:
        """bin 바인딩 및 스케일 갱신."""
        self.bin = bin_obj
        # (width, height, depth)
        self._sx_top   = self.W / float(self.bin.width)
        self._sy_top   = self.H / float(self.bin.height)

        self._sx_front = self.W / float(self.bin.width)
        self._sy_front = self.H / float(self.bin.depth)

        self._sx_left  = self.W / float(self.bin.height)
        self._sy_left  = self.H / float(self.bin.depth)

        if self.use_fast2d:
            self.reset_buffers()

    def reset_buffers(self) -> None:
        """증분 버퍼 초기화."""
        H, W = self.H, self.W
        def _blank():
            occ   = np.zeros((H, W), dtype=np.uint16)
            shade = np.full((H, W), 255, dtype=np.uint8)
            img   = np.repeat(shade[..., None], 3, axis=2).copy()
            return occ, shade, img

        self._occ_top,   self._shade_top,   self._img_top   = _blank()
        self._occ_front, self._shade_front, self._img_front = _blank()
        self._occ_left,  self._shade_left,  self._img_left  = _blank()

    def rebuild_from_bin(self) -> None:
        """전체 아이템으로 버퍼 재구성."""
        if not self.use_fast2d:
            return
        self.reset_buffers()
        for it in self.bin.get_all_items():
            self.update_with_item(it)

    def update_with_item(self, it) -> None:
        """아이템 1개 증분 반영."""
        if not self.use_fast2d:
            return

        # TOP (width, height)
        x0, x1, y0, y1 = self._coords_top_from_item(it)
        if x1 > x0 and y1 > y0:
            sub = self._occ_top[y0:y1, x0:x1]
            np.add(sub, 1, out=sub, casting="unsafe")
            shade = self._shade_from_occ(sub)
            self._shade_top[y0:y1, x0:x1] = shade
            self._img_top[y0:y1, x0:x1, :] = shade[:, :, None]

        # FRONT (width, depth)
        x0, x1, y0, y1 = self._coords_front_from_item(it)
        if x1 > x0 and y1 > y0:
            sub = self._occ_front[y0:y1, x0:x1]
            np.add(sub, 1, out=sub, casting="unsafe")
            shade = self._shade_from_occ(sub)
            self._shade_front[y0:y1, x0:x1] = shade
            self._img_front[y0:y1, x0:x1, :] = shade[:, :, None]

        # LEFT (height, depth)
        x0, x1, y0, y1 = self._coords_left_from_item(it)
        if x1 > x0 and y1 > y0:
            sub = self._occ_left[y0:y1, x0:x1]
            np.add(sub, 1, out=sub, casting="unsafe")
            shade = self._shade_from_occ(sub)
            self._shade_left[y0:y1, x0:x1] = shade
            self._img_left[y0:y1, x0:x1, :] = shade[:, :, None]

    def get_triplet_hwc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (top, left, front)의 HWC[uint8] 반환.
        fast2d → 버퍼 그대로 사용
        engine → top/left/front 순으로 시도, left/front 실패 시 간단 투영 대체
        """
        if self.use_fast2d:
            return self._img_top, self._img_left, self._img_front

        top   = self._render_topdown_rgb()               # top:(width,height)
        left  = self._render_view_rgb("left")            # 가능하면 엔진
        front = self._render_view_rgb("front")

        # 실패 시 간단 투영으로 대체
        if left is None:
            left = self._fast2d_from_bin_plane("left")
        if front is None:
            front = self._fast2d_from_bin_plane("front")
        return top, left, front

    # ---------- private: shade/coords ----------
    def _shade_from_occ(self, occ_sub: np.ndarray) -> np.ndarray:
        if self.fast2d_mode == "count":
            return (255 - 40 * occ_sub).clip(0, 255).astype(np.uint8)
        elif self.fast2d_mode == "alpha":
            return (255.0 * ((1.0 - 0.35) ** occ_sub)).astype(np.uint8)
        else:
            return (255 - 40 * occ_sub).clip(0, 255).astype(np.uint8)

    def _coords_top_from_item(self, it):
        x, y, z = it.b_position    # x=width, y=height, z=depth
        w, h, d = it.getDimension()
        x0 = max(0, min(self.W, int(x       * self._sx_top)))
        x1 = max(0, min(self.W, int((x + w) * self._sx_top)))
        y0 = max(0, min(self.H, int(y       * self._sy_top)))
        y1 = max(0, min(self.H, int((y + h) * self._sy_top)))
        return x0, x1, y0, y1

    def _coords_front_from_item(self, it):
        """front: (width, depth) — x↦width, y↦depth"""
        x, y, z = it.b_position
        w,  h,  d  = it.getDimension()
        x0 = max(0, min(self.W, int( x      * self._sx_front )))
        x1 = max(0, min(self.W, int((x + w) * self._sx_front )))
        y0 = max(0, min(self.H, int( z      * self._sy_front )))
        y1 = max(0, min(self.H, int((z + d) * self._sy_front )))
        return x0, x1, y0, y1

    def _coords_left_from_item(self, it):
        """left: (height, depth) — x↦height, y↦depth"""
        x, y, z = it.b_position
        w,  h,  d  = it.getDimension()
        x0 = max(0, min(self.W, int( y      * self._sx_left )))
        x1 = max(0, min(self.W, int((y + h) * self._sx_left )))
        y0 = max(0, min(self.H, int( z      * self._sy_left )))
        y1 = max(0, min(self.H, int((z + d) * self._sy_left )))
        return x0, x1, y0, y1

    # ---------- private: engine/fast2d fallback ----------
    def _render_topdown_rgb(self) -> np.ndarray:
        """
        (width, height) 평면을 보는 천정 뷰.
        bin.render의 topdown=True가 elev=90, azim=-90, proj='ortho'로 강제됨.
        """
        try:
            rgb = self.bin.render(
                show=False, save=False, return_array=True,
                write_num=False,
                topdown=True  # ← 핵심
            )
        except Exception:
            rgb = None

        if rgb is None:
            rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        im  = Image.fromarray(rgb).resize((self.W, self.H), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim == 2:  # grayscale → RGB
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 4:  # RGBA → RGB
            arr = arr[..., :3]
        return arr


    def _render_view_rgb(self, view: str) -> np.ndarray | None:
        """
        front: (width, depth)
        left : (height, depth)
        – ortho(정사영)로 고정해 2D 평면 정보만 보이게.
        """
        if view not in ("front", "left"):
            return None

        if view == "front":
            view_elev, view_azim = 0, -90  # +y(높이) 방향에서 바라봄 → (x,z) 평면
        else:  # "left"
            view_elev, view_azim = 0, 0    # +x(너비) 방향에서 바라봄 → (y,z) 평면

        try:
            rgb = self.bin.render(
                show=False, save=False, return_array=True, write_num=False,
                topdown=False,                     # ← topdown은 사용하지 않음
                view_elev=view_elev, view_azim=view_azim,
                proj_type="ortho"                  # ← 원근 왜곡 제거
            )
        except Exception:
            rgb = None

        if rgb is None:
            return None

        im  = Image.fromarray(rgb).resize((self.W, self.H), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr


    def _fast2d_from_bin_plane(self, plane: str) -> np.ndarray:
        """
        주어진 평면으로 2D 정사영을 빠르게 생성 (HWC uint8).
        plane ∈ {"top", "front", "left"}
        - top   : (width,  height) = (x, y)
        - front : (width,  depth ) = (x, z)
        - left  : (height, depth ) = (y, z)
        """
        H, W = self.H, self.W
        occ = np.zeros((H, W), dtype=np.uint16)

        # 스케일 & 좌표 선택
        if plane == "top":
            sx = W / float(self.bin.width)
            sy = H / float(self.bin.height)
            def rect(it):
                bx, by, _ = it.b_position
                w,  h,  _ = it.getDimension()
                x0 = max(0, min(W, int(bx * sx)))
                x1 = max(0, min(W, int((bx + w) * sx)))
                y0 = max(0, min(H, int(by * sy)))
                y1 = max(0, min(H, int((by + h) * sy)))
                return x0, x1, y0, y1

        elif plane == "front":
            sx = W / float(self.bin.width)
            sy = H / float(self.bin.depth)
            def rect(it):
                bx, _, bz = it.b_position
                w,  _,  d = it.getDimension()
                x0 = max(0, min(W, int(bx * sx)))
                x1 = max(0, min(W, int((bx + w) * sx)))
                y0 = max(0, min(H, int(bz * sy)))
                y1 = max(0, min(H, int((bz + d) * sy)))
                return x0, x1, y0, y1

        elif plane == "left":
            sx = W / float(self.bin.height)
            sy = H / float(self.bin.depth)
            def rect(it):
                _, by, bz = it.b_position
                _,  h,  d = it.getDimension()
                x0 = max(0, min(W, int(by * sx)))
                x1 = max(0, min(W, int((by + h) * sx)))
                y0 = max(0, min(H, int(bz * sy)))
                y1 = max(0, min(H, int((bz + d) * sy)))
                return x0, x1, y0, y1

        else:
            raise ValueError(f"unknown plane: {plane}")

        # 모든 아이템 정사영 누적 (겹치면 카운트+1)
        for it in self.bin.get_all_items():
            x0, x1, y0, y1 = rect(it)
            if x1 > x0 and y1 > y0:
                occ[y0:y1, x0:x1] += 1

        # 점유 → 그레이스케일
        occ_i = occ.astype(np.int32)
        if self.fast2d_mode == "count":
            shade = np.clip(255 - 40 * occ_i, 0, 255).astype(np.uint8)
        elif self.fast2d_mode == "alpha":
            shade = (255.0 * ((1.0 - 0.35) ** occ_i)).astype(np.uint8)
        else:
            shade = np.clip(255 - 40 * occ_i, 0, 255).astype(np.uint8)

        # HWC 3채널로 반환
        return np.repeat(shade[..., None], 3, axis=2)

    def ensure_chw_u8(self, img: np.ndarray) -> np.ndarray:
        # dtype 맞추기
        if img.dtype != np.uint8:
            # float [0,1] or [0,255] 가 들어와도 안전하게 u8로
            scale = 255.0 if img.max() <= 1.001 else 1.0
            img = np.clip(img * scale, 0, 255).astype(np.uint8)

        # 레이아웃 맞추기: HWC(…, …, 3) -> CHW(3, …, …)
        if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
            img = np.transpose(img, (2, 0, 1))
        return img

    def chw_to_hwc_u8(self, img_chw: np.ndarray) -> np.ndarray:
        # 디버그 저장 용
        assert img_chw.ndim == 3 and img_chw.shape[0] == 3
        return np.transpose(img_chw, (1, 2, 0))