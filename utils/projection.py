# utils/projection.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def project_point_to_plane(pt, plane):
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)
    t = (np.dot(n, pt) + d) / np.dot(n, n)
    return pt - t * n

def _choose_axes(a, b, c):
    if abs(a) >= abs(b) and abs(a) >= abs(c):
        return 0
    elif abs(b) >= abs(a) and abs(b) >= abs(c):
        return 1
    return 2

def _liang_barsky(u0, v0, u1, v1, umin, umax, vmin, vmax):
    p = [-(u1-u0),  (u1-u0),  -(v1-v0),  (v1-v0)]
    q = [u0-umin,   umax-u0,   v0-vmin,   vmax-v0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        r = qi / pi
        if pi < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return t0, t1

def boundary_projection(face_plane, face_bounds, p0, p1, atol=1e-6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    p0p = project_point_to_plane(p0, face_plane)
    p1p = project_point_to_plane(p1, face_plane)

    a, b, c, _ = face_plane
    drop = _choose_axes(a, b, c)

    coord_map = (
        (lambda p: (p[1], p[2]), ('y', 'z')),
        (lambda p: (p[0], p[2]), ('x', 'z')),
        (lambda p: (p[0], p[1]), ('x', 'y'))
    )[drop]

    (u0, v0), (u1, v1) = coord_map[0](p0p), coord_map[0](p1p)
    u_axis, v_axis = coord_map[1]
    umin, umax = face_bounds[u_axis]
    vmin, vmax = face_bounds[v_axis]

    clip = _liang_barsky(u0, v0, u1, v1, umin, umax, vmin, vmax)
    if clip is None:
        return None

    t0, t1 = clip                      # 0 ≤ t0 ≤ t1 ≤ 1
    line_vec = p1p - p0p
    p_clip0  = p0p + line_vec * t0     # 잘린 구간의 시작
    p_clip1  = p0p + line_vec * t1     # 잘린 구간의 끝

    # t0==t1→점 하나만 남은 경우, 그렇지 않으면 두 점 반환
    if abs(t1 - t0) < atol:
        return p_clip0.tolist()
    return [p_clip0.tolist(), p_clip1.tolist()]

# helper: build rectangle vertices for a face (reused for multiple faces)
def build_rect(face_plane, face_bounds):
    a, b, c, d = face_plane
    # detect which axis is fixed by checking a,b,c
    if abs(a) > 0:   # plane is x = const
        x_fixed = face_bounds['x'][0]
        y_min, y_max = face_bounds['y']
        z_min, z_max = face_bounds['z']
        rect = [(x_fixed, y_min, z_min),
                (x_fixed, y_max, z_min),
                (x_fixed, y_max, z_max),
                (x_fixed, y_min, z_max)]
    elif abs(b) > 0: # plane is y = const
        y_fixed = face_bounds['y'][0]
        x_min, x_max = face_bounds['x']
        z_min, z_max = face_bounds['z']
        rect = [(x_min, y_fixed, z_min),
                (x_max, y_fixed, z_min),
                (x_max, y_fixed, z_max),
                (x_min, y_fixed, z_max)]
    else:            # plane is z = const
        z_fixed = face_bounds['z'][0]
        x_min, x_max = face_bounds['x']
        y_min, y_max = face_bounds['y']
        rect = [(x_min, y_min, z_fixed),
                (x_max, y_min, z_fixed),
                (x_max, y_max, z_fixed),
                (x_min, y_max, z_fixed)]
    return rect

if __name__ == "__main__":
    # ---------------- inputs ----------------
    # Plane A: y = 100
    face_plane_A  = (0.0, 1.0, 0.0, -100.0)
    face_bounds_A = {'x': (50.0, 150.0), 'y': (100.0, 100.0), 'z': (100.0, 220.0)}

    # Plane B: x = 250
    face_plane_B  = (0.0, 1.0, 0.0, -100.0)
    face_bounds_B = {'x': (200.0, 300.0), 'y': (100.0, 100.0),  'z': (100.0, 220.0)}

    # line endpoints
    p0 = np.array([100., 200., 170.])
    p1 = np.array([250., 200., 170.])

    # ---------------- compute projections for both planes ----------------
    proj_A = boundary_projection(face_plane_A, face_bounds_A, p0, p1)
    proj_B = boundary_projection(face_plane_B, face_bounds_B, p0, p1)
    print("Projected on A:", proj_A)
    print("Projected on B:", proj_B)

    p0p_A = project_point_to_plane(p0, face_plane_A)
    p1p_A = project_point_to_plane(p1, face_plane_A)
    p0p_B = project_point_to_plane(p0, face_plane_B)
    p1p_B = project_point_to_plane(p1, face_plane_B)

    # ---------------- build rects ----------------
    rect_A = build_rect(face_plane_A, face_bounds_A)
    rect_B = build_rect(face_plane_B, face_bounds_B)

    # ----------------- visualization -----------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # optional: remove ticks / grid for a clean view
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)

    # draw planes (different colors)
    polyA = Poly3DCollection([rect_A], alpha=0.15)
    polyA.set_facecolor('gray')
    ax.add_collection3d(polyA)

    polyB = Poly3DCollection([rect_B], alpha=0.18)
    polyB.set_facecolor('gray')
    ax.add_collection3d(polyB)

    # original line + endpoints (thick black)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='black', linewidth=3)
    ax.scatter([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
               color='black', s=40, depthshade=False, marker='o')

    # projected onto A (red)
    if proj_A is not None:
        # dashed projected line (plane A)
        ax.plot([p0p_A[0], p1p_A[0]], [p0p_A[1], p1p_A[1]], [p0p_A[2], p1p_A[2]],
                color='red', linestyle='--', linewidth=2)

        # projected clipped points (plane A)
        proj_pts_A = proj_A if isinstance(proj_A[0], (list, tuple, np.ndarray)) else [proj_A]
        xsA, ysA, zsA = zip(*proj_pts_A)
        ax.scatter(xsA, ysA, zsA,
                   c='red', marker='o', s=80, edgecolors='k', linewidths=0.6, depthshade=False, alpha=1.0)
        if len(proj_pts_A) == 2:
            ax.plot(xsA, ysA, zsA, color='red', linewidth=3)

    # projected onto B (blue)
    if proj_B is not None:
        ax.plot([p0p_B[0], p1p_B[0]], [p0p_B[1], p1p_B[1]], [p0p_B[2], p1p_B[2]],
                color='red', linestyle='--', linewidth=2)

        proj_pts_B = proj_B if isinstance(proj_B[0], (list, tuple, np.ndarray)) else [proj_B]
        xsB, ysB, zsB = zip(*proj_pts_B)
        ax.scatter(xsB, ysB, zsB,
                   c='red', marker='o', s=80, edgecolors='k', linewidths=0.6, depthshade=False, alpha=1.0)
        if len(proj_pts_B) == 2:
            ax.plot(xsB, ysB, zsB, color='red', linewidth=3)

    # final axes labels & limits
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(0, 400); ax.set_ylim(0, 300); ax.set_zlim(0, 250)
    plt.tight_layout()
    plt.show()
