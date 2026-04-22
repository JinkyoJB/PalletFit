# utils/painter/painter_plot.py
import matplotlib
# matplotlib.use('TkAgg')  # 🚀 GUI 창을 띄울 수 있도록 강제 설정
matplotlib.use('Agg')  # 화면 출력 없이 파일로 저장하는 백엔드
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
import mpl_toolkits.mplot3d.art3d as art3d
from typing import Optional, List, Tuple
import numpy as np
import io
import os
# from planning.itemManager import global_item_manager


class PainterPlot:

    def __init__(self, bin):
        ''' '''
        self.width = bin.width
        self.height = bin.height
        self.depth = bin.depth
        self.bin_name = bin.name
        self.bin = bin

    def _plotCube(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        """ Auxiliary function to plot a cube. code taken somewhere from the web.  """
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]

        kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text != "":
                ax.text((x + dx / 2), (y + dy / 2), (z + dz / 2), str(text), color='black', fontsize=fontsize, ha='center', va='center')

            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")

    def _plotCylinder(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, text="", fontsize=10, alpha=0.2):
        """ Auxiliary function to plot a Cylinder  """
        # plot the two circles above and below the cylinder
        p = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        p2 = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
        # plot a circle in the middle of the cylinder
        center_z = np.linspace(0, dz, 10)
        theta = np.linspace(0, 2*np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, center_z)
        x_grid = dx / 2 * np.cos(theta_grid) + x + dx / 2
        y_grid = dy / 2 * np.sin(theta_grid) + y + dy / 2
        z_grid = z_grid + z
        ax.plot_surface(x_grid, y_grid, z_grid, shade=False, fc=color, alpha=alpha, color=color)
        if text != "":
            ax.text((x + dx / 2), (y + dy / 2), (z + dz / 2), str(text), color='black', fontsize=fontsize, ha='center', va='center')


    def plotBoxAndItems(
            self,
            title="",
            alpha=0.2,
            write_num=False,
            fontsize=10,
            save=False,
            show=False,
            save_path="planning/renders",
            draw_composite=True,
            size_annotation: bool = False,
            show_pivots=True,
            *,
            save_dpi=300,
            view_elev: float = 60,
            view_azim: float = -30,
            view_roll=None,
            view_dist=None,
            proj_type: str = "persp",
            pivot_point_size=5,
            pivots: Optional[List[Tuple[float, float, float]]] = None,
            pivot_mark_size: int = 40,
            pivot_mark_color: str = "lime",
            it_ids: Optional[List[int]] = None,
            return_array: bool = False,
        ):
        """
        Bin + Items (+ optional pivotTree) 3D 그림.
        - return_array=True: (H,W,3) uint8 ndarray 반환, 제목/축/범례/장식 제거 + 여백 0, bin이 프레임 꽉 차게.
        - return_array=False: 기존 동작(파일/표시/PNG bytes).
        """
        self._set_backend(show)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 카메라/투영
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
        if view_dist is not None:
            ax.dist = view_dist
        if proj_type == "ortho":
            ax.set_proj_type("ortho")

        # ── Bin wireframe
        self._plotCube(ax, 0, 0, 0,
                    float(self.width), float(self.height), float(self.depth),
                    color="black", mode=1, linewidth=2)

        # ── Items
        for item in self.bin.get_all_items():
            if not draw_composite and getattr(item, "is_composite", False):
                continue
            x, y, z = item.b_position
            w, h, d = item.getDimension()
            color = getattr(item, "color", "gray") or getattr(item, "options", {}).get("color", "gray")
            # it_id가 리스트이거나 단일값일 수 있으므로 모두 처리
            if it_ids is not None:
                if isinstance(it_ids, (list, tuple, set)):
                    if item._id in it_ids:
                        color = 'red'
                else:
                    if item._id == it_ids:
                        color = 'red'
            if len(item.children_ids) > 0:
                for child_id in item.children_ids:
                    if child_id == it_ids:
                        color = 'red'
                        break
            if size_annotation:
                label = f"{w:.1f}x{h:.3f}x{d:.1f}"
            else:
                label = f"{item.name}_{item._id}" if write_num else ""
            if item.objshape == "cube":
                self._plotCube(ax, x, y, z, w, h, d,
                            color=color, mode=2,
                            text=label, fontsize=fontsize, alpha=alpha)
            elif item.objshape == "cylinder":
                self._plotCylinder(ax, x, y, z, w, h, d,
                                color=color, mode=2,
                                text=label, fontsize=fontsize, alpha=alpha)

        # ── Pivot 시각화 (배열 반환 모드에선 범례 등 장식 제거)
        if show_pivots is True and self.bin.pivotTree is not None and not return_array:
            # 1. 색상 매핑 수정: 2버전도 원본과 같은 색으로 변경
            dir_color = {
                "left": "red",    "left2": "red",      # left2 -> red (left와 동일)
                "right": "blue",  "right2": "blue",    # right2 -> blue (일관성 유지)
                "front": "orange","front2": "orange",  # front2 -> orange (front와 동일)
                "back": "purple",
                "down": "green", "up": "cyan",
                "any": "black", "cp": "blue", "ep": "blue", "ems": "blue",
            }
            legend_handles = {}
            
            for pv in self.bin.pivotTree.in_order_traversal():
                raw_dir: str = getattr(pv, "direction", "any")
                key_dir = raw_dir.split("-")[0]
                
                # 2. 라벨 그룹화 로직 추가 ('left2' -> 'left')
                if key_dir.endswith("2"): 
                    label_key = key_dir[:-1]  # 끝의 '2' 제거
                else:
                    label_key = key_dir

                col = dir_color.get(key_dir, "black")
                
                ax.scatter(pv.x, pv.y, pv.z, c=col, s=pivot_point_size, depthshade=False)
                
                if write_num:
                    ax.text(pv.x, pv.y, pv.z, label_key[0].upper(), color=col, fontsize=6, ha="center")
                
                # 3. 범례 생성 시 통합된 label_key 사용
                if label_key not in legend_handles:
                    handle = ax.scatter([], [], [], c=col, s=30, label=label_key)
                    legend_handles[label_key] = handle
            
            ax.legend(handles=list(legend_handles.values()),
                    title="Pivot direction", loc="upper right", fontsize=8)

        if pivots and not return_array:
            for idx, (px, py, pz) in enumerate(pivots):
                ax.scatter(px, py, pz, c=pivot_mark_color, s=pivot_mark_size,
                        marker='*', edgecolors='black', linewidths=0.6, depthshade=False)
                if write_num:
                    ax.text(px, py, pz, f"P{idx}", color=pivot_mark_color,
                            fontsize=8, ha="center", va="center")

        # ── 축/라벨/화살표: 배열 반환 모드에서는 모두 생략
        if not return_array:
            ax.set_xlabel("X", fontsize=11)
            ax.set_ylabel("Y", fontsize=11)
            ax.set_zlabel("Z", fontsize=11)
            arrow_kw = dict(color="black", linewidth=1.2, arrow_length_ratio=0.05)
            ax.quiver(0, 0, 0, self.width*0.8, 0, 0, **arrow_kw)   # X
            ax.quiver(0, 0, 0, 0, self.depth*0.8, 0, **arrow_kw)   # Y
            ax.quiver(0, 0, 0, 0, 0, self.height*0.8, **arrow_kw)  # Z

        # ── 레이아웃: bin이 꽉 차게 / 여백 제거
        ax.set_box_aspect([self.width, self.height, self.depth])
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, self.depth)
        ax.margins(0)

        if not return_array:
            plt.title(title)

        # === 반환 로직 ===
        if return_array:
            # ① 모든 장식 제거 + axes를 figure 전체로 확장 (여백 0)
            ax.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1)
            ax.set_position([0, 0, 1, 1])

            # ② ortho + 줌인: dist를 줄이면 더 꽉 차 보입니다 (기본 10 → 4)
            try:
                if getattr(ax, "dist", None) is not None:
                    ax.dist = 4  # 필요시 3~6 사이로 조절해보세요
            except Exception:
                pass

            # ③ 캔버스에서 직접 RGB 추출
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            rgba_mem = canvas.buffer_rgba()
            arr = np.frombuffer(rgba_mem, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()

            # ④ 흰 가장자리 자동 크롭 (pad 몇 픽셀 남김)
            arr = self._crop_rgb_to_content(arr, pad=4)

            # (선택) 파일 저장/표시
            if save:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/{title or 'snap'}.jpg",
                            format="jpg", dpi=save_dpi, bbox_inches="tight", pad_inches=0)
            if show:
                plt.show(block=True)

            plt.close(fig)
            return arr  # (H, W, 3) uint8
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            if save:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(f"{save_path}/{title or 'snap'}.jpg",
                            format="jpg", dpi=save_dpi, bbox_inches="tight", pad_inches=0)
            if show:
                plt.show(block=True)
            plt.close(fig)
            return buf.getvalue()  # PNG bytes

    def _crop_rgb_to_content(self, arr: np.ndarray, pad: int = 4) -> np.ndarray:
        """
        흰 배경(≈255) 기준으로 유효 픽셀 바운딩박스를 찾아 크롭.
        pad: 가장자리에 조금 여유 두기 (픽셀 단위)
        """
        if arr.ndim != 3 or arr.shape[2] != 3:
            return arr
        # 흰색(255)에 아주 가까운 영역은 배경으로 간주
        gray = arr.mean(axis=2)
        mask = gray < 250  # 255보다 살짝 낮게 threshold
        if not mask.any():
            return arr
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h, w = arr.shape[:2]
        y0 = max(y0 - pad, 0); y1 = min(y1 + pad, h - 1)
        x0 = max(x0 - pad, 0); x1 = min(x1 + pad, w - 1)
        return arr[y0:y1+1, x0:x1+1]
    
    def _set_backend(self, show):
        """Set the appropriate matplotlib backend based on `show`."""
        if show:
            try:
                matplotlib.use('TkAgg')  # Use a GUI backend for showing plots
            except ImportError:
                print("Warning: TkAgg backend not available. Falling back to 'agg'.")
                matplotlib.use('agg')
        else:
            matplotlib.use('agg')

    def setAxesEqual(self, ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    def plotZonesAndItems(self, zones, title="", alpha=0.2, write_num=False, fontsize=10, save=False, show=False):
        """ zones와 items를 함께 plot하는 메서드 
            zones: zones 리스트
        """
        # Set backend based on `show`
        self._set_backend(show)
        # figure와 axes를 명시적으로 생성
        fig = plt.figure()
        axGlob = fig.add_subplot(111, projection='3d')
        axGlob.view_init(elev=60, azim=-30)  # 뷰 설정
        
        # bin을 wireframe으로 표시
        self._plotCube(axGlob, 0, 0, 0, float(self.width), float(self.height), float(self.depth), 
                    color='black', mode=1, linewidth=1, text="")

        # ✅ Zones 표시
        for i, zone in enumerate(zones):
            zone_color = 'green'
            x, y, z = zone.x, zone.y, zone.z
            w, h, d = zone.width, zone.height, zone.depth
            
            # leftover zone이면 alpha를 더 높게 하여 강조
            zone_alpha = alpha * 0.3
            text_str = f"{zone.name}"  # zone에 번호 부여

            # zone cube plot
            self._plotCube(
                axGlob, float(x), float(y), float(z),
                float(w), float(h), float(d),
                color=zone_color, mode=2,
                text=text_str if write_num else "",
                fontsize=fontsize, alpha=zone_alpha
            )

        # ✅ Items 표시
        counter = 0
        for item in self.get_all_items():
            x, y, z = item.b_position
            w, h, d = item.getDimension()


            if item.options.get('is_attached') is True:
                color = 'red'

            text = item.name +'_' + str(item._id )if write_num else ""

            if item.objshape == 'cube':
                self._plotCube(axGlob, float(x), float(y), float(z), float(w), float(h), float(d),
                            color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.objshape == 'cylinder':
                self._plotCylinder(axGlob, float(x), float(y), float(z), float(w), float(h), float(d),
                                color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)

            counter += 1

        plt.title(title)
        axGlob.set_box_aspect([self.width, self.height, self.depth])  # aspect ratio is 1:1:1
        axGlob.margins(0)

        # ✅ 메모리에 이미지 저장
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
        img_buffer.seek(0)  # 버퍼를 처음 위치로 되돌림

        if save:
            save_path = 'planning/renders'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f'{save_path}/zone_{title}.jpg', format='jpg', bbox_inches='tight', pad_inches=0)

        if show:
            # plt.figure()
            # plt.imshow(plt.imread(img_buffer))  # 🚀 저장한 이미지 로드하여 새 창에 표시
            # plt.axis("off")
            plt.show(block=True) 

        plt.close(fig)

        return img_buffer.getvalue()  # ✅ GUI로 보낼 수 있도록 이미지 데이터 반환