import numpy as np
from typing import Dict, Tuple

def _rng(seed=None):
    return np.random.default_rng(seed)

def sample_uniform_rect(N: int, xlim: Tuple[float,float], ylim: Tuple[float,float], seed=None) -> np.ndarray:
    r = _rng(seed)
    x = r.uniform(xlim[0], xlim[1], size=(N, 1))
    y = r.uniform(ylim[0], ylim[1], size=(N, 1))
    return np.hstack([x, y])

def sample_line_x(N: int, x_fixed: float, ylim: Tuple[float,float], seed=None) -> np.ndarray:
    r = _rng(seed)
    y = r.uniform(ylim[0], ylim[1], size=(N, 1))
    x = np.full((N, 1), float(x_fixed))
    return np.hstack([x, y])

def sample_line_y(N: int, y_fixed: float, xlim: Tuple[float,float], seed=None) -> np.ndarray:
    r = _rng(seed)
    x = r.uniform(xlim[0], xlim[1], size=(N, 1))
    y = np.full((N, 1), float(y_fixed))
    return np.hstack([x, y])

def sample_near_boundary(N: int, boundary_xy: np.ndarray, band: float = 0.02, seed=None) -> np.ndarray:
    """
    Sample points near the airfoil boundary by jittering boundary points.
    band controls Gaussian std-dev of jitter.
    """
    r = _rng(seed)
    idx = r.integers(0, boundary_xy.shape[0], size=N)
    base = boundary_xy[idx]
    jitter = r.normal(loc=0.0, scale=band, size=base.shape)
    return base + jitter

def remove_inside_and_attach_d(xy: np.ndarray, airfoil) -> np.ndarray:
    """
    airfoil must have:
      - airfoil.is_inside(xy)->bool mask
      - airfoil.distance(xy)->(N,1)
    Returns array [x,y,d] for points outside airfoil.
    """
    inside = airfoil.is_inside(xy)
    xy_out = xy[~inside]
    d = airfoil.distance(xy_out)
    return np.hstack([xy_out, d])

def build_point_cloud(
    airfoil,
    xlim=(-1.0, 2.0),
    ylim=(-0.5, 0.5),
    N_int=30000,
    N_inlet=4000,
    N_outlet=4000,
    N_top=3000,
    N_bot=3000,
    N_airfoil=4000,
    N_near=15000,
    near_band=0.02,
    seed=1234,
) -> Dict[str, np.ndarray]:
    """
    Returns dictionary of point sets, each is array with columns [x,y,d].
    """
    r = _rng(seed)

    # interior points
    xy_int = sample_uniform_rect(N_int, xlim, ylim, seed=r.integers(1e9))
    X_int = remove_inside_and_attach_d(xy_int, airfoil)

    # farfield boundaries (rectangle)
    xL, xR = xlim
    yB, yT = ylim

    xy_inlet  = sample_line_x(N_inlet,  xL, (yB, yT), seed=r.integers(1e9))
    xy_outlet = sample_line_x(N_outlet, xR, (yB, yT), seed=r.integers(1e9))
    xy_top    = sample_line_y(N_top,    yT, (xL, xR), seed=r.integers(1e9))
    xy_bot    = sample_line_y(N_bot,    yB, (xL, xR), seed=r.integers(1e9))

    # airfoil boundary points (use boundary itself, subsample)
    b = airfoil.boundary
    if b.shape[0] >= N_airfoil:
        idx = r.choice(b.shape[0], size=N_airfoil, replace=False)
        xy_air = b[idx]
    else:
        idx = r.integers(0, b.shape[0], size=N_airfoil)
        xy_air = b[idx]

    # near-airfoil band points (jitter boundary)
    xy_near = sample_near_boundary(N_near, b, band=near_band, seed=r.integers(1e9))

    # attach d (and remove inside for near points)
    X_inlet  = remove_inside_and_attach_d(xy_inlet, airfoil)
    X_outlet = remove_inside_and_attach_d(xy_outlet, airfoil)
    X_top    = remove_inside_and_attach_d(xy_top, airfoil)
    X_bot    = remove_inside_and_attach_d(xy_bot, airfoil)

    # airfoil boundary points are on the wall => distance should be ~0
    d_air = airfoil.distance(xy_air)
    X_air = np.hstack([xy_air, d_air])

    X_near = remove_inside_and_attach_d(xy_near, airfoil)

    return {
        "interior": X_int,
        "inlet": X_inlet,
        "outlet": X_outlet,
        "top": X_top,
        "bottom": X_bot,
        "airfoil": X_air,
        "near": X_near,
    }
