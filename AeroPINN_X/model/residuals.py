import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SAConstants:
    sigma: float = 2/3
    cb1: float = 0.1355
    cb2: float = 0.622
    kappa: float = 0.41
    cw2: float = 0.3
    cw3: float = 2.0
    cv1: float = 7.1

    @property
    def cw1(self) -> float:
        return self.cb1/(self.kappa**2) + (1 + self.cb2)/self.sigma

def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )[0]

def safe_pos(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.clamp(x, min=eps)

def clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)

def sa_turbulent_viscosity(nu_tilde: torch.Tensor, nu: torch.Tensor, c: SAConstants) -> torch.Tensor:
    chi = nu_tilde / safe_pos(nu)
    chi = clamp(chi, 0.0, 1e6)
    chi3 = chi**3
    fv1 = chi3 / (chi3 + c.cv1**3)
    return nu_tilde * fv1

def pde_residuals_rans_sa(x_in: torch.Tensor, y_out: torch.Tensor, sa: SAConstants = SAConstants()) -> dict:
    if not x_in.requires_grad:
        raise ValueError("x_in must have requires_grad=True BEFORE model forward: x.requires_grad_(True); y=model(x)")

    d = x_in[:, 3:4]
    d_safe = safe_pos(d, eps=1e-4)          # IMPORTANT: stronger floor for stability

    # x_in[:,4] is log10(Re_phys)
    Re_log10 = x_in[:, 4:5]
    Re = 10.0 ** Re_log10
    nu = 1.0 / safe_pos(Re)

    u = y_out[:, 0:1]
    v = y_out[:, 1:2]
    p = y_out[:, 2:3]

    # enforce nu_tilde >= 0
    nu_tilde_raw = y_out[:, 3:4]
    nu_tilde = F.softplus(nu_tilde_raw) + 1e-8

    nu_t = sa_turbulent_viscosity(nu_tilde, nu, sa)
    nu_eff = nu + nu_t

    du = grad(u, x_in)
    dv = grad(v, x_in)
    dp = grad(p, x_in)
    dnut = grad(nu_tilde, x_in)

    u_x = du[:, 0:1]; u_y = du[:, 1:2]
    v_x = dv[:, 0:1]; v_y = dv[:, 1:2]
    p_x = dp[:, 0:1]; p_y = dp[:, 1:2]
    nut_x = dnut[:, 0:1]; nut_y = dnut[:, 1:2]

    u_xx = grad(u_x, x_in)[:, 0:1]
    u_yy = grad(u_y, x_in)[:, 1:2]
    v_xx = grad(v_x, x_in)[:, 0:1]
    v_yy = grad(v_y, x_in)[:, 1:2]

    nut_xx = grad(nut_x, x_in)[:, 0:1]
    nut_yy = grad(nut_y, x_in)[:, 1:2]

    # continuity
    r_cont = u_x + v_y

    # momentum (simple RANS diffusion form)
    nu_eff_x = grad(nu_eff, x_in)[:, 0:1]
    nu_eff_y = grad(nu_eff, x_in)[:, 1:2]

    diff_u = nu_eff * (u_xx + u_yy) + nu_eff_x * u_x + nu_eff_y * u_y
    diff_v = nu_eff * (v_xx + v_yy) + nu_eff_x * v_x + nu_eff_y * v_y

    r_mom_u = u * u_x + v * u_y + p_x - diff_u
    r_mom_v = u * v_x + v * v_y + p_y - diff_v

    # ---------- SA (stabilized) ----------
    omega = v_x - u_y
    S = torch.abs(omega)

    chi = nu_tilde / safe_pos(nu)
    chi = clamp(chi, 0.0, 1e6)
    chi3 = chi**3
    fv1 = chi3 / (chi3 + sa.cv1**3)
    fv2 = 1.0 - chi / safe_pos(1.0 + chi * fv1)

    S_tilde = S + (nu_tilde / safe_pos(sa.kappa**2 * d_safe**2)) * fv2
    S_tilde = safe_pos(S_tilde, eps=1e-6)

    # production
    P = sa.cb1 * S_tilde * nu_tilde

    # destruction with clamps
    r_var = nu_tilde / (S_tilde * (sa.kappa**2) * d_safe**2 + 1e-12)
    r_var = clamp(r_var, 0.0, 10.0)

    g = r_var + sa.cw2 * (r_var**6 - r_var)
    g = clamp(g, 0.0, 10.0)

    fw = g * ((1.0 + sa.cw3**6) / safe_pos(g**6 + sa.cw3**6))**(1.0/6.0)
    fw = clamp(fw, 0.0, 5.0)

    D = sa.cw1 * fw * (nu_tilde**2) / safe_pos(d_safe**2)

    # diffusion
    nu_hat = safe_pos(nu + nu_tilde, eps=1e-8)
    nu_hat_x = grad(nu_hat, x_in)[:, 0:1]
    nu_hat_y = grad(nu_hat, x_in)[:, 1:2]

    div_term = nu_hat * (nut_xx + nut_yy) + nu_hat_x * nut_x + nu_hat_y * nut_y
    gradnut2 = nut_x**2 + nut_y**2
    Diff = (1.0 / sa.sigma) * (div_term + sa.cb2 * gradnut2)

    r_sa = u * nut_x + v * nut_y - (P - D + Diff)

    return {
        "cont": r_cont,
        "mom_u": r_mom_u,
        "mom_v": r_mom_v,
        "sa": r_sa,
        "nu_t": nu_t,
        "nu_eff": nu_eff,
        "nu_tilde_pos": nu_tilde
    }
