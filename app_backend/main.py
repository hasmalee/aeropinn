from __future__ import annotations

import os
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---- your project imports ----
from AeroPINN_X.geometry.airfoil import AirfoilGeometry
from AeroPINN_X.data.sampling import build_point_cloud
from AeroPINN_X.model.network import MLP
from AeroPINN_X.train.train_loop import train
from AeroPINN_X.model.residuals import pde_residuals_rans_sa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

OPTIMIZATIONS = {}
APP_ROOT = Path(__file__).resolve().parents[1]          
ARTIFACTS_DIR = APP_ROOT / "artifacts"    #ARTIFACTS_DIR = Path("artifacts")            
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) #ARTIFACTS_DIR.mkdir(exist_ok=True)

RUNS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="AeroPINN-X Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- helpers ----------------
def _plot_point_cloud(run_dir: str, cloud: dict, boundary):
    import matplotlib.pyplot as plt

    # Accept both names just in case (backward compatible)
    interior = cloud.get("interior", None)
    if interior is None:
        interior = cloud.get("int", None)  # legacy fallback

    near = cloud.get("near", None)

    plt.figure(figsize=(8, 3))

    if interior is not None:
        plt.scatter(interior[:, 0], interior[:, 1], s=1, label="interior", alpha=0.5)

    if near is not None:
        plt.scatter(near[:, 0], near[:, 1], s=2, label="near", alpha=0.7)

    plt.plot(boundary[:, 0], boundary[:, 1], "-", lw=2, label="airfoil")
    plt.axis("equal")
    plt.legend(loc="lower right")
    plt.title("Point cloud (mesh-free)")

    out = os.path.join(run_dir, "point_cloud.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out



def _predict_on_grid(model, air: AirfoilGeometry, alpha_deg: float, Re_phys: float, device: str):
    # simple rectangular grid
    xs = np.linspace(-1.0, 2.0, 220)
    ys = np.linspace(-0.5, 0.5, 120)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    d = air.distance(pts).astype(np.float32)  # (N,1) or (N,)
    if d.ndim == 1:
        d = d[:, None]

    alpha_col = np.full((pts.shape[0], 1), alpha_deg, dtype=np.float32)
    Re_col = np.full((pts.shape[0], 1), Re_phys, dtype=np.float32)

    x_in = np.concatenate([pts, alpha_col, d, Re_col], axis=1)
    xt = torch.from_numpy(x_in).to(device)

    with torch.no_grad():
        y = model(xt)
    u = y[:, 0:1].cpu().numpy()
    v = y[:, 1:2].cpu().numpy()
    p = y[:, 2:3].cpu().numpy()
    nu = y[:, 3:4].cpu().numpy()
    speed = np.sqrt(u**2 + v**2)

    # reshape to grid
    u = u.reshape(Y.shape)
    v = v.reshape(Y.shape)
    p = p.reshape(Y.shape)
    nu = nu.reshape(Y.shape)
    speed = speed.reshape(Y.shape)
    return X, Y, u, v, p, nu, speed


# def _contour_plot(run_dir: Path, X, Y, field, title: str, fname: str) -> None:
#     plt.figure(figsize=(7, 4))
#     cf = plt.contourf(X, Y, field, levels=60)
#     plt.colorbar(cf)
#     plt.title(title)
#     plt.axis("equal")
#     plt.tight_layout()
#     plt.savefig(run_dir / fname, dpi=200)
#     plt.close()
def _contour_plot(run_dir: str, X, Y, Z, title: str, fname: str, boundary=None):
    Zp = Z.copy()

    # 1) mask inside airfoil polygon (so it becomes a “solid body” hole)
    if boundary is not None:
        boundary = np.asarray(boundary)
        poly = MplPath(boundary)
        pts = np.c_[X.ravel(), Y.ravel()]
        inside = poly.contains_points(pts).reshape(Z.shape)
        Zp[inside] = np.nan

    # 2) plot contours
    plt.figure(figsize=(8, 3))
    plt.contourf(X, Y, Zp, levels=60)     # more levels = smoother like paper
    plt.colorbar()
    plt.axis("equal")
    plt.title(title)

    # 3) draw airfoil outline on top
    if boundary is not None:
        plt.plot(boundary[:, 0], boundary[:, 1], "k-", lw=2)

    out = os.path.join(run_dir, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _train_and_score_one_alpha(
    run_dir: Path,
    airfoil_path: Path,
    alpha_deg: float,
    Re_phys: float,
    steps: int,
    N_int: int,
    N_near: int,
    lr: float,
) -> tuple:
    """
    Train a model for a single angle of attack and return objective + artifacts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # geometry
    air = AirfoilGeometry(str(airfoil_path), n_boundary=1200)
    boundary = air.boundary
    
    # cloud
    cloud = build_point_cloud(
        airfoil=air,
        xlim=(-1.0, 2.0),
        ylim=(-0.5, 0.5),
        N_int=N_int,
        N_near=N_near,
    )
    _plot_point_cloud(str(run_dir), cloud, boundary)
    
    # train
    model = MLP(in_dim=5, out_dim=4, hidden_dim=128, num_hidden_layers=6, activation="tanh").to(device)
    model = train(
        model=model,
        airfoil=air,
        alpha_deg=alpha_deg,
        Re_phys=Re_phys,
        N_int=N_int,
        N_near=N_near,
        steps=steps,
        lr=lr,
        print_every=max(10, steps // 10),
        save_every=10**9,
        out_dir=str(run_dir),
        device=device,
    )
    
    # inference
    X, Y, u, v, p, nu, speed = _predict_on_grid(model, air, alpha_deg, Re_phys, device=device)
    _contour_plot(str(run_dir), X, Y, u, "u velocity", "u.png", boundary=air.boundary)
    _contour_plot(str(run_dir), X, Y, v, "v velocity", "v.png", boundary=air.boundary)
    _contour_plot(str(run_dir), X, Y, p, "pressure", "pressure.png", boundary=air.boundary)
    _contour_plot(str(run_dir), X, Y, speed, "Speed magnitude √(u²+v²)", "speed.png", boundary=air.boundary)
    
    # objective: maximize pressure coefficient (or minimize drag proxy)
    objective = float(np.mean(speed))  # or another metric
    
    artifacts = {
        "speed": f"speed.png",
        "pressure": f"pressure.png",
    }
    
    return objective, artifacts


def _prem_heatmap(run_dir: Path, model, xt: torch.Tensor):
    # residuals at points xt
    y = model(xt)
    res = pde_residuals_rans_sa(xt, y)
    cont = res["cont"].detach().abs().cpu().numpy().reshape(-1)
    sa = res["sa"].detach().abs().cpu().numpy().reshape(-1)

    x = xt.detach().cpu().numpy()
    Xp, Yp = x[:, 0], x[:, 1]

    def save_scatter(vals, fname, title):
        plt.figure(figsize=(7, 4))
        sc = plt.scatter(Xp, Yp, c=vals, s=3)
        plt.colorbar(sc)
        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(run_dir / fname, dpi=200)
        plt.close()

    save_scatter(cont, "prem_cont.png", "PREM: continuity residual |cont|")
    save_scatter(sa, "prem_sa.png", "PREM: SA residual |sa|")

    stats = {
        "cont_mean": float(cont.mean()),
        "cont_max": float(cont.max()),
        "sa_mean": float(sa.mean()),
        "sa_max": float(sa.max()),
    }
    return stats


def _run_training_job(run_id: str) -> None:
    """
    IMPORTANT: this function takes ONLY run_id.
    All configs are read from RUNS[run_id]["config"].
    """
    try:
        RUNS[run_id]["status"] = "running"
        cfg = RUNS[run_id]["config"]
        run_dir: Path = RUNS[run_id]["run_dir"]
        airfoil_path: Path = RUNS[run_id]["airfoil_path"]

        alpha = float(cfg["alpha_deg"])
        Re = float(cfg["Re_phys"])
        steps = int(cfg["steps"])
        N_int = int(cfg["N_int"])
        N_near = int(cfg["N_near"])
        lr = float(cfg["lr"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        RUNS[run_id]["device"] = device

        # geometry
        air = AirfoilGeometry(str(airfoil_path), n_boundary=1200)
        boundary = air.boundary

        # cloud
        cloud = build_point_cloud(
            airfoil=air,
            xlim=(-1.0, 2.0),
            ylim=(-0.5, 0.5),
            N_int=N_int,
            N_near=N_near,
        )
        _plot_point_cloud(run_dir, cloud, boundary)

        # train (prototype)
        model = MLP(in_dim=5, out_dim=4, hidden_dim=128, num_hidden_layers=6, activation="tanh").to(device)
        model = train(
            model=model,
            airfoil=air,
            alpha_deg=alpha,
            Re_phys=Re,
            N_int=N_int,
            N_near=N_near,
            steps=steps,
            lr=lr,
            print_every=max(10, steps // 10),
            save_every=10**9,
            out_dir=str(run_dir),  # <-- use run_dir instead of None
            device=device,
        )

        # inference
        X, Y, u, v, p, nu, speed = _predict_on_grid(model, air, alpha, Re, device=device)
        _contour_plot(str(run_dir), X, Y, u, "u velocity", "u.png", boundary=air.boundary)
        _contour_plot(str(run_dir), X, Y, v, "v velocity", "v.png", boundary=air.boundary)
        _contour_plot(str(run_dir), X, Y, p, "pressure", "pressure.png", boundary=air.boundary)
        _contour_plot(str(run_dir), X, Y, speed, "Speed magnitude √(u²+v²)", "speed.png", boundary=air.boundary)

        print("u min/max:", u.min(), u.max())
        print("v min/max:", v.min(), v.max())
        print("p min/max:", p.min(), p.max())

        # PREM (near points)
        near = cloud["near"]
        d = air.distance(near[:, :2])
        if d.ndim == 1:
            d = d[:, None]
        x_in = np.concatenate(
            [
                near[:, :2].astype(np.float32),
                np.full((near.shape[0], 1), alpha, dtype=np.float32),
                d.astype(np.float32),
                np.full((near.shape[0], 1), Re, dtype=np.float32),
            ],
            axis=1,
        )
        xt = torch.from_numpy(x_in).to(device).requires_grad_(True)
        prem_stats = _prem_heatmap(run_dir, model, xt)

        RUNS[run_id]["status"] = "done"
        RUNS[run_id]["result"] = {
            "run_id": run_id,
            "status": "done",
            "device": device,
            "prem": prem_stats,
            "artifacts": {
                "point_cloud": f"/artifacts/{run_id}/point_cloud.png",
                "speed": f"/artifacts/{run_id}/speed.png",
                "pressure": f"/artifacts/{run_id}/pressure.png",
                "prem_cont": f"/artifacts/{run_id}/prem_cont.png",
                "prem_sa": f"/artifacts/{run_id}/prem_sa.png",
            },
        }

    except Exception as e:
        RUNS[run_id]["status"] = "error"
        RUNS[run_id]["error"] = repr(e) + "\n" + traceback.format_exc()

def _run_aoa_optimization_job(opt_id: str):
    try:
        OPTIMIZATIONS[opt_id]["status"] = "running"

        cfg = OPTIMIZATIONS[opt_id]["config"]
        Re = float(cfg["Re_phys"])
        alpha_min = float(cfg["alpha_min"])
        alpha_max = float(cfg["alpha_max"])
        n_alpha = int(cfg["n_alpha"])

        steps = int(cfg["steps"])
        N_int = int(cfg["N_int"])
        N_near = int(cfg["N_near"])
        lr = float(cfg["lr"])

        opt_dir = Path(OPTIMIZATIONS[opt_id]["opt_dir"])
        airfoil_path = Path(OPTIMIZATIONS[opt_id]["airfoil_path"])

        alphas = np.linspace(alpha_min, alpha_max, n_alpha).tolist()

        rows = []
        best = None

        for i, alpha in enumerate(alphas):
            # Reuse your existing training job pipeline, but in a "sub-run" folder
            sub_run_id = f"{opt_id}_a{i}"
            sub_dir = opt_dir / sub_run_id
            sub_dir.mkdir(parents=True, exist_ok=True)

            # ---- CALL YOUR EXISTING PIPELINE HERE ----
            # You already have logic like:
            # air = AirfoilGeometry(...)
            # cloud = build_point_cloud(...)
            # model = train(...)
            # predict fields -> speed,p, etc.
            #
            # For now, create a helper that runs one alpha and returns objective + artifacts.
            obj, artifacts = _train_and_score_one_alpha(
                run_dir=sub_dir,
                airfoil_path=airfoil_path,
                alpha_deg=float(alpha),
                Re_phys=Re,
                steps=steps,
                N_int=N_int,
                N_near=N_near,
                lr=lr,
            )

            rows.append({
                "alpha": float(alpha),
                "objective": float(obj),
                "sub_run_id": sub_run_id,
                "artifacts": artifacts,
            })

            if (best is None) or (obj > best["objective"]):
                best = rows[-1]

        OPTIMIZATIONS[opt_id]["status"] = "done"
        OPTIMIZATIONS[opt_id]["result"] = {
            "opt_id": opt_id,
            "status": "done",
            "Re_phys": Re,
            "alphas": alphas,
            "table": rows,
            "best": best,
        }

    except Exception as e:
        OPTIMIZATIONS[opt_id]["status"] = "error"
        OPTIMIZATIONS[opt_id]["error"] = repr(e)


# ---------------- API ----------------
@app.post("/run_async")
async def run_async(
    background_tasks: BackgroundTasks,
    airfoil_dat: UploadFile = File(...),                 # must match frontend key "airfoil_dat" :contentReference[oaicite:0]{index=0}
    alpha_deg: float = Form(5.0),
    Re_phys: float = Form(1e6),
    steps: int = Form(100),
    N_int: int = Form(2000),
    N_near: int = Form(2000),
    lr: float = Form(1e-4),
    checkpoint_path: Optional[str] = Form(None),
):
    run_id = uuid.uuid4().hex[:10]
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # save uploaded file
    contents = await airfoil_dat.read()
    airfoil_path = run_dir / airfoil_dat.filename
    airfoil_path.write_bytes(contents)

    RUNS[run_id] = {
        "status": "queued",
        "run_dir": run_dir,
        "airfoil_path": airfoil_path,
        "config": {
            "alpha_deg": alpha_deg,
            "Re_phys": Re_phys,
            "steps": steps,
            "N_int": N_int,
            "N_near": N_near,
            "lr": lr,
            "checkpoint_path": checkpoint_path,
        },
        "result": None,
        "error": None,
    }

    # IMPORTANT: pass only run_id (fixes “takes 1 positional argument but 9 were given”)
    background_tasks.add_task(_run_training_job, run_id)

    return {"run_id": run_id, "status": "queued"}


@app.get("/runs/{run_id}/status")
def run_status(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found (server restarted?)")
    r = RUNS[run_id]
    return {"run_id": run_id, "status": r["status"], "error": r.get("error")}


@app.get("/runs/{run_id}/result")
def run_result(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found (server restarted?)")
    r = RUNS[run_id]
    if r["status"] != "done":
        raise HTTPException(status_code=409, detail=f"run not done (status={r['status']})")
    return r["result"]


@app.get("/artifacts/{run_id}/{filename}")
def get_artifact(run_id: str, filename: str):
    p = ARTIFACTS_DIR / run_id / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(str(p))

@app.post("/optimize_aoa")
async def optimize_aoa(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    Re_phys: float = Form(...),
    alpha_min: float = Form(...),
    alpha_max: float = Form(...),
    n_alpha: int = Form(...),
    steps: int = Form(...),
    N_int: int = Form(...),
    N_near: int = Form(...),
    lr: float = Form(...),
):
    opt_id = uuid.uuid4().hex[:10]
    opt_dir = ARTIFACTS_DIR / f"opt_{opt_id}"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # save uploaded airfoil
    airfoil_path = opt_dir / file.filename
    content = await file.read()
    airfoil_path.write_bytes(content)

    OPTIMIZATIONS[opt_id] = {
        "status": "queued",
        "error": None,
        "result": None,
        "opt_dir": str(opt_dir),
        "airfoil_path": str(airfoil_path),
        "config": {
            "Re_phys": Re_phys,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "n_alpha": n_alpha,
            "steps": steps,
            "N_int": N_int,
            "N_near": N_near,
            "lr": lr,
        },
    }

    # IMPORTANT: pass a single argument (opt_id) to avoid your earlier “takes 1 arg but 9 were given” issue
    background_tasks.add_task(_run_aoa_optimization_job, opt_id)

    return {"opt_id": opt_id, "status": "started"}

@app.get("/optimize/{opt_id}/status")
def optimize_status(opt_id: str):
    if opt_id not in OPTIMIZATIONS:
        return {"status": "not_found"}
    d = OPTIMIZATIONS[opt_id]
    return {"opt_id": opt_id, "status": d["status"], "error": d["error"]}

@app.get("/optimize/{opt_id}/result")
def optimize_result(opt_id: str):
    if opt_id not in OPTIMIZATIONS:
        return {"status": "not_found"}
    d = OPTIMIZATIONS[opt_id]
    if d["status"] != "done":
        return {"status": d["status"], "error": d["error"]}
    return d["result"]
    
@app.get("/health")
def health():
    return {"status": "ok"}
