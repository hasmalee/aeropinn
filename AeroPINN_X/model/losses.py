# (file content)
import torch
from AeroPINN_X.model.residuals import pde_residuals_rans_sa

def _mse(x: torch.Tensor) -> torch.Tensor:
    return (x**2).mean()

def total_loss(cloud, model,
               w_pde=1.0, w_near=1.0,
               w_inlet=10.0, w_outlet=10.0, w_far=10.0,
               w_wall=50.0):
    """
    cloud: dict of torch tensors with keys:
      'interior', 'near', 'inlet', 'outlet', 'top', 'bottom', 'airfoil'
    each tensor shape: (N,5) = [x,y,alpha,d,Re_log10] with requires_grad=True
    model: network
    returns dict of losses (torch scalars)
    """

    # ---------------- PDE losses ----------------
    # interior PDE
    x_int = cloud["interior"]
    y_int = model(x_int)
    res_int = pde_residuals_rans_sa(x_int, y_int)

    L_pde = _mse(res_int["cont"]) + _mse(res_int["mom_u"]) + _mse(res_int["mom_v"]) + _mse(res_int["sa"])

    # near-wall PDE (helps turbulent wall region)
    x_near = cloud["near"]
    y_near = model(x_near)
    res_near = pde_residuals_rans_sa(x_near, y_near)

    L_near = _mse(res_near["cont"]) + _mse(res_near["mom_u"]) + _mse(res_near["mom_v"]) + _mse(res_near["sa"])

    # ---------------- Boundary condition losses ----------------
    # Freestream target (prototype)
    u_inf = 1.0
    v_inf = 0.0
    nu_inf = 1e-6

    # inlet
    xin = cloud["inlet"]
    yin = model(xin)
    uin, vin, pin, nuin = yin[:,0:1], yin[:,1:2], yin[:,2:3], yin[:,3:4]
    L_inlet = _mse(uin - u_inf) + _mse(vin - v_inf) + _mse(nuin - nu_inf)

    # outlet: pressure reference
    xout = cloud["outlet"]
    yout = model(xout)
    pout = yout[:,2:3]
    L_outlet = _mse(pout - 0.0)

    # top and bottom treated as farfield
    xtop = cloud["top"]
    ytop = model(xtop)
    ut, vt, pt, nut = ytop[:,0:1], ytop[:,1:2], ytop[:,2:3], ytop[:,3:4]
    L_top = _mse(ut - u_inf) + _mse(vt - v_inf) + _mse(nut - nu_inf)

    xbot = cloud["bottom"]
    ybot = model(xbot)
    ub, vb, pb, nub = ybot[:,0:1], ybot[:,1:2], ybot[:,2:3], ybot[:,3:4]
    L_bottom = _mse(ub - u_inf) + _mse(vb - v_inf) + _mse(nub - nu_inf)

    # wall (airfoil): no-slip and nu_tilde=0
    xw = cloud["airfoil"]
    yw = model(xw)
    uw, vw, pw, nuw = yw[:,0:1], yw[:,1:2], yw[:,2:3], yw[:,3:4]
    L_wall = _mse(uw - 0.0) + _mse(vw - 0.0) + _mse(nuw - 0.0)

    # ---------------- Total ----------------
    total = (
        w_pde * L_pde +
        w_near * L_near +
        w_inlet * L_inlet +
        w_outlet * L_outlet +
        w_far * (L_top + L_bottom) +
        w_wall * L_wall
    )

    return {
        "total": total,
        "pde": L_pde,
        "near": L_near,
        "inlet": L_inlet,
        "outlet": L_outlet,
        "top": L_top,
        "bottom": L_bottom,
        "wall": L_wall,
    }
