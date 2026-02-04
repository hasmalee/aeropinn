import os
import math
import torch
import numpy as np

from AeroPINN_X.data.sampling import build_point_cloud
from AeroPINN_X.model.losses import total_loss

def _add_params(X_xyd: torch.Tensor, alpha_deg: float, Re_phys: float, device: str):
    """
    X_xyd: (N,3) = [x,y,d]
    returns (N,5) = [x,y,alpha(rad),d,log10(Re)]
    """
    alpha = math.radians(alpha_deg)
    Re_log10 = math.log10(Re_phys)

    N = X_xyd.shape[0]
    alpha_col = torch.full((N,1), alpha, dtype=torch.float32, device=device)
    Re_col    = torch.full((N,1), Re_log10, dtype=torch.float32, device=device)
    x_in = torch.cat([X_xyd[:,0:2], alpha_col, X_xyd[:,2:3], Re_col], dim=1)
    x_in.requires_grad_(True)
    return x_in

def train(
    model,
    airfoil,
    alpha_deg: float = 5.0,
    Re_phys: float = 1e6,
    xlim=(-1.0, 2.0),
    ylim=(-0.5, 0.5),
    N_int: int = 20000,
    N_near: int = 20000,
    steps: int = 500,
    lr: float = 1e-4,
    print_every: int = 50,
    save_every: int = 250,
    out_dir: str = "/content/drive/MyDrive/AeroPINN_X_SE/checkpoints",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps+1):
        cloud_np = build_point_cloud(
            airfoil=airfoil,
            xlim=xlim, ylim=ylim,
            N_int=N_int, N_near=N_near,
            N_airfoil=4000, N_inlet=4000, N_outlet=4000, N_top=3000, N_bot=3000,
            near_band=0.02,
            seed=step
        )

        # Convert numpy -> torch and add params
        cloud = {}
        for k, arr in cloud_np.items():
            X = torch.tensor(arr, dtype=torch.float32, device=device)  # (N,3) [x,y,d]
            cloud[k] = _add_params(X, alpha_deg=alpha_deg, Re_phys=Re_phys, device=device)

        opt.zero_grad(set_to_none=True)
        losses = total_loss(cloud=cloud, model=model)
        L = losses["total"]

        if not torch.isfinite(L):
            print("STOP: loss is non-finite at step", step, "loss:", L)
            break

        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % print_every == 0:
            print(f"step {step}/{steps} | total {losses['total'].item():.3e} | "
                  f"pde {losses['pde'].item():.3e} | near {losses['near'].item():.3e} | "
                  f"wall {losses['wall'].item():.3e}")

        if step % save_every == 0:
            ckpt = os.path.join(out_dir, f"pinn_step_{step}.pt")
            torch.save({"step": step, "model_state": model.state_dict()}, ckpt)
            print("saved:", ckpt)

    return model
