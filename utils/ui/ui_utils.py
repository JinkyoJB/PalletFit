from pyqtgraph.opengl import MeshData, GLViewWidget, GLLinePlotItem, GLMeshItem, GLGridItem
import numpy as np, pyqtgraph as pg



# ─── GLBinViewer (라벨 제거판) ─────────────────────────────────
class GLBinViewer(GLViewWidget):
    """bin wire + 3-면 grid + item mesh (라벨 없음)"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.setBackgroundColor((220, 220, 220, 30))
        self._wire, self._grids, self._items = None, {}, {}
        # 축 표시
        for vec, col in [((1,0,0),'r'), ((0,1,0),'g'), ((0,0,1),'b')]:
            pts = np.array([[0,0,0], np.multiply(vec, 600)])
            self.addItem(GLLinePlotItem(pos=pts, color=col, width=2))

    # ---- 외부에서 호출 ------------------------------------------------
    def update_scene(self, payload: dict):
        meta, items = payload['bin'], payload['items']
        w,h,d = meta['width'], meta['height'], meta['depth']
        self._draw_bin_wire(w,h,d)
        # self._ensure_grids(w,h,d)
        self._draw_items(items)
        self.opts['center'] = pg.Vector(w/2, h/2, d/2)
        self.setCameraPosition(
            distance = max(w, h, d) * 3,
            elevation = 25,      # 위쪽 각도는 그대로
            azimuth   = 30 + 180 # == 210°, 뒤쪽에서 같은 각도로 바라보기
        )

    # ---- bin wire ----------------------------------------------------
    def _draw_bin_wire(self, w,h,d):
        pts = np.array([[0,0,0],[w,0,0],[w,h,0],[0,h,0],[0,0,0],
                        [0,0,d],[w,0,d],[w,h,d],[0,h,d],[0,0,d],
                        [w,0,d],[w,0,0],[w,h,0],[w,h,d],[0,h,d],[0,h,0]])
        if self._wire is None:
            self._wire = GLLinePlotItem(pos=pts, color=(1,1,1,1), width=1)
            self.addItem(self._wire)
        else:
            self._wire.setData(pos=pts)

    # ---- 3-면 grid ---------------------------------------------------
    def _ensure_grids(self, w,h,d, spacing=100):
        DARK = (0.25,0.25,0.25,1)
        def _grid(key, sizeXY, transforms):
            g = self._grids.get(key) or GLGridItem()
            g.setSpacing(spacing, spacing); g.setSize(*sizeXY); g.setColor(DARK)
            g.setGLOptions('opaque'); g.resetTransform()
            for fn, args in transforms: getattr(g, fn)(*args)
            if key not in self._grids: self.addItem(g); self._grids[key]=g

        _grid('floor', (w, h), [('translate',(w/2,h/2,0))])
        _grid('front', (w, d), [('rotate',(90,1,0,0)), ('translate',(w/2,0,d/2))])
        _grid('left' , (h, d), [('rotate',(90,0,1,0)), ('translate',(0,h/2,d/2))])

    # ---- item meshes -------------------------------------------------
    def _draw_items(self, items):
        alive = {it['id'] for it in items}
        for it in items:
            iid = it['id']; r,g,b,_ = pg.mkColor(it['color']).getRgbF()
            color = (1,0,0,0.5) if it.get('is_target') else (r,g,b,0.4)
            mesh = self._items.get(iid)
            if mesh is None:
                mesh = GLMeshItem(meshdata=_unit_cube_mesh(),
                                  smooth=False, drawEdges=True,
                                  edgeColor=(0,0,0,0.4))
                mesh.setGLOptions('translucent'); self.addItem(mesh)
                self._items[iid] = mesh
            mesh.setColor(color)
            mesh.resetTransform()
            mesh.scale(it['w'], it['h'], it['d'])
            mesh.translate(it['x']+it['w']/2, it['y']+it['h']/2, it['z']+it['d']/2)
        # 제거
        for iid in list(self._items):
            if iid not in alive: self.removeItem(self._items.pop(iid))

# ─── 유틸: 단위 큐브 ───────────────────────────────────────────
def _unit_cube_mesh() -> MeshData:
    v = np.array([[-.5,-.5,-.5],[ .5,-.5,-.5],[ .5, .5,-.5],[-.5, .5,-.5],
                  [-.5,-.5, .5],[ .5,-.5, .5],[ .5, .5, .5],[-.5, .5, .5]])
    f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [1,2,6],[1,6,5],[3,0,4],[3,4,7]])
    return MeshData(vertexes=v, faces=f)