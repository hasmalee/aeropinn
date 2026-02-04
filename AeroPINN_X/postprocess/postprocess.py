# AeroPINN_X/postprocess/postprocess.py
import numpy as np
import torch

from AeroPINN_X.model.residuals import pde_residuals_rans_sa

def make_grid(xlim=(-1.0, 2.0), ylim=(-0.5, 0.5), nx=240, ny=140):
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return X, Y, pts

@torch.no_grad()
def predict_on_points(model, pts_xy, alpha_deg, Re_phys, airfoil, device="cpu"):
    """
    pts_xy: (N,2) numpy points in domain
    Build x=[x,y,alpha,d,log10(Re)] and predict y=[u,v,p,nu_tilde_raw]
    """
    model.eval()
    pts_xy = np.asarray(pts_xy, dtype=np.float32)

    d = airfoil.distance(pts_xy).astype(np.float32).reshape(-1,1)
    alpha = np.full((pts_xy.shape[0],1), alpha_deg, dtype=np.float32)
    re_log = np.full((pts_xy.shape[0],1), np.log10(Re_phys), dtype=np.float32)

    x = np.concatenate([pts_xy, alpha, d, re_log], axis=1)  # (N,5)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = model(xt)  # (N,4)
    y = yt.detach().cpu().numpy()
    return x, y

def predict_fields_and_residuals(model, alpha_deg, Re_phys, airfoil,
                                xlim=(-1,2), ylim=(-0.5,0.5),
                                nx=240, ny=140, device="cpu"):
    """
    Returns grid + fields + residual heatmaps on the grid.
    """
    X, Y, pts = make_grid(xlim, ylim, nx, ny)

    # fields (no grad)
    x_in, y_out = predict_on_points(model, pts, alpha_deg, Re_phys, airfoil, device=device)
    u = y_out[:,0].reshape(Y.shape)
    v = y_out[:,1].reshape(Y.shape)
    p = y_out[:,2].reshape(Y.shape)
    nut = y_out[:,3].reshape(Y.shape)
    Vmag = np.sqrt(u*u + v*v)

    # residuals (need grad)
    xt = torch.tensor(x_in, dtype=torch.float32, device=device, requires_grad=True)
    yt = model(xt)
    res = pde_residuals_rans_sa(xt, yt)
    cont = res["cont"].detach().cpu().numpy().reshape(Y.shape)
    sa   = res["sa"].detach().cpu().numpy().reshape(Y.shape)

    out = {
        "X": X, "Y": Y,
        "u": u, "v": v, "p": p, "nu_tilde": nut, "Vmag": Vmag,
        "res_cont": cont,
        "res_sa": sa,
    }
    return out

def cp_on_boundary(model, airfoil, alpha_deg, Re_phys, n=400, device="cpu"):
    """
    Cp curve along resampled airfoil boundary.
    For prototype: assume nondim rho=1, U_inf=1, p_inf=0 => Cp = p / 0.5 = 2p
    (If your nondimensionalization differs, adjust here.)
    """
    b = airfoil.boundary
    if b.shape[0] > n:
        idx = np.linspace(0, b.shape[0]-1, n).astype(int)
        b = b[idx]

    x_in, y_out = predict_on_points(model, b, alpha_deg, Re_phys, airfoil, device=device)
    p = y_out[:,2:3]  # (n,1)

    Cp = (p / 0.5).reshape(-1)  # prototype scaling
    xcoord = b[:,0].reshape(-1)
    ycoord = b[:,1].reshape(-1)
    return xcoord, ycoord, Cp
