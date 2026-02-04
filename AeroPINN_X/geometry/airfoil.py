import numpy as np
from scipy.spatial import cKDTree

def read_dat(dat_path: str) -> np.ndarray:
    pts = []
    with open(dat_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if any(c.isalpha() for c in s):
                continue
            a = s.split()
            if len(a) >= 2:
                try:
                    pts.append([float(a[0]), float(a[1])])
                except ValueError:
                    pass

    pts = np.array(pts, dtype=float)
    if pts.shape[0] < 20:
        raise ValueError("Too few points read from .dat")

    # remove numeric "count line" like '66 66' if present at the top
    if pts.shape[0] >= 5:
        first = pts[0]
        rest = pts[1:]
        if (abs(first[0]) > 5 and abs(first[1]) > 5) and (np.max(np.abs(rest)) < 5):
            pts = rest

    # scale if points are in percent chord (0..100)
    p95 = np.percentile(np.abs(pts), 95)
    if p95 > 5.0:
        pts = pts / 100.0

    # drop any remaining outliers
    mask = (pts[:,0] > -2.0) & (pts[:,0] < 3.0) & (np.abs(pts[:,1]) < 2.0)
    pts = pts[mask]

    return pts

def resample_closed_curve(pts: np.ndarray, n: int = 1200) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
        pts = np.vstack([pts, pts[0]])

    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt((seg**2).sum(axis=1))
    s = np.hstack([[0.0], np.cumsum(seglen)])
    L = s[-1]
    if L <= 0:
        raise ValueError("Invalid curve length")

    t = np.linspace(0, L, n, endpoint=False)
    out = np.zeros((n, 2), dtype=float)

    j = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        ds = s[j + 1] - s[j]
        w = 0.0 if ds < 1e-14 else (ti - s[j]) / ds
        out[i] = (1 - w) * pts[j] + w * pts[j + 1]

    return out

def point_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    poly = np.asarray(poly, dtype=float)

    if np.linalg.norm(poly[0] - poly[-1]) > 1e-10:
        poly = np.vstack([poly, poly[0]])

    x = xy[:, 0]
    y = xy[:, 1]
    xp = poly[:, 0]
    yp = poly[:, 1]

    inside = np.zeros(len(xy), dtype=bool)
    for i in range(len(poly) - 1):
        x1, y1 = xp[i], yp[i]
        x2, y2 = xp[i + 1], yp[i + 1]
        cond = ((y1 > y) != (y2 > y))
        x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-20) + x1
        inside ^= cond & (x < x_int)
    return inside

class AirfoilGeometry:
    def __init__(self, dat_path: str, n_boundary: int = 1200):
        raw = read_dat(dat_path)
        self.boundary = resample_closed_curve(raw, n=n_boundary)
        self._kdtree = cKDTree(self.boundary)

    def distance(self, xy: np.ndarray) -> np.ndarray:
        d, _ = self._kdtree.query(np.asarray(xy, dtype=float), k=1)
        return d.reshape(-1, 1)

    def is_inside(self, xy: np.ndarray) -> np.ndarray:
        return point_in_polygon(np.asarray(xy, dtype=float), self.boundary)
