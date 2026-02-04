import requests
import streamlit as st
import base64
import streamlit as st
from pathlib import Path
import os


st.set_page_config(page_title="AeroPINN-X", layout="wide")

# ---- sidebar controls ----
# prefer env var BACKEND_URL for deployments (Railway); still allow manual override in the sidebar
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND, key="backend_url")

page = st.sidebar.radio(
    "Page",
    ["Home", "Upload + Preview", "Train", "Results", "PREM Heatmaps", "optimize AOA"],
    key="page_select"
)

alpha = st.sidebar.slider("AoA (deg)", -5.0, 15.0, 5.0, 0.5, key="alpha_slider")
Re_phys = st.sidebar.selectbox("Re (physical)", [1e5, 5e5, 1e6, 3e6], index=2, key="re_select")
steps = st.sidebar.slider("Training steps", 50, 2000, 100, 50, key="steps_slider")
N_int = st.sidebar.selectbox("N_int", [2000, 4000, 8000, 20000], index=0, key="nint_select")
N_near = st.sidebar.selectbox("N_near", [2000, 4000, 8000, 20000], index=0, key="nnear_select")
lr = st.sidebar.selectbox("Learning rate", [1e-3, 5e-4, 1e-4, 5e-5], index=2, key="lr_select")
checkpoint_path = st.sidebar.text_input("Checkpoint path (optional)", "", key="checkpoint_path_input")

# ---- session state ----
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "airfoil_bytes" not in st.session_state:
    st.session_state.airfoil_bytes = None
if "airfoil_name" not in st.session_state:
    st.session_state.airfoil_name = None
if "opt_id" not in st.session_state:
    st.session_state.opt_id = None
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None

def api_get(path, timeout=30):
    r = requests.get(f"{BACKEND}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post_run(file_bytes: bytes, file_name: str):
    files = {
        "airfoil_dat": (file_name, file_bytes, "text/plain")
    }
    data = {
        "alpha_deg": str(alpha),
        "Re_phys": str(Re_phys),
        "steps": str(steps),
        "N_int": str(N_int),
        "N_near": str(N_near),
        "lr": str(lr),
        "checkpoint_path": checkpoint_path or "",
    }

    r = requests.post(f"{BACKEND}/run_async", files=files, data=data, timeout=3600)

    # If it fails, show the real FastAPI error message (super important)
    if r.status_code == 422:
        st.error(r.text)
        return None

    r.raise_for_status()
    return r.json()
def api_post_optimize_aoa(
    file_bytes: bytes,
    file_name: str,
    Re_phys: float,
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
    steps: int,
    N_int: int,
    N_near: int,
    lr: float,
):
    files = {"file": (file_name, file_bytes, "text/plain")}
    data = {
        "Re_phys": str(Re_phys),
        "alpha_min": str(alpha_min),
        "alpha_max": str(alpha_max),
        "n_alpha": str(n_alpha),
        "steps": str(steps),
        "N_int": str(N_int),
        "N_near": str(N_near),
        "lr": str(lr),
    }

    r = requests.post(f"{BACKEND}/optimize_aoa", files=files, data=data, timeout=3600)

    if r.status_code == 422:
        st.error(r.text)
        return None

    r.raise_for_status()
    return r.json()
def set_bg_image(path: str):
    img_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Optional: make main content readable on top of image */
        section[data-testid="stMain"] > div {{
            background-color: rgba(255, 255, 255, 0.88);
            border-radius: 16px;
            padding: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("app_frontend/assets/22.jpg")

def render_home():
    st.caption("Fast aerodynamic field prediction + AoA optimization (no CFD solver).")

    # Simple hero box
    st.markdown(
        """
        <div style="padding:16px;border-radius:12px;background:#f6f8ff;border:1px solid #e6e9ff;">
            <h3 style="margin:0;">Your design tool for High-Re airfoil exploration</h3>
            <p style="margin:6px 0 0 0;">
                AeroPINN-X turns early airfoil design questions into fast, visual, physics-checked 
                insights letting you test ideas, compare conditions, and run quick AoA optimization 
                without waiting for full CFD.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About")
        st.write(
            """
            AeroPINN-X is an engineering prototype built with a Physics-Informed Neural Network (PINN).
            It learns to satisfy the governing equations and boundary conditions, so we can approximate
            aerodynamic flow fields without a traditional CFD solver in the loop.
            """
        )

        st.subheader("Designed for")
        st.markdown(

            """
            - Conceptual / preliminary design where you need rapid iteration (low-Re scenarios, UAVs, small wings, research validation)
            - Quick “what-if” studies across AoA and Re (compare trends, not days of simulation time)
            - Demonstrating a complete product workflow: Upload → Train → Results → Residual QA → Optimization
            - Students/researchers who want physics + software engineering in one deployable tool 
            """
        )

    with col2:
        st.subheader("Key features")
        st.markdown(
            """
            - Upload airfoil `.dat`  
            - Point cloud preview (interior + near-wall points)  
            - Prototype training (short runs with live status + saved artifacts)  
            - Output plots: **u, v, p, ν̃**  
            - PREM residual heatmaps (continuity + SA) to verify physics
            - AoA optimization loop (objective vs iteration)  
            """
        )

        st.subheader("How to use (demo flow)")
        st.markdown(
            """
            1) **Upload + Preview** → upload `.dat`, confirm shape / points  
            2) **Train** → start async run (prototype steps)  
            3) **Results** → view predicted fields and plots  
            4) **PREM Heatmaps** → validate physics residuals 
            5) **optimize AOA** → run AoA sweep → show objective curve → pick best AoA
            """
        )

    st.write("")
    st.info("Tip: For quick demos, use small steps and moderate N_int/N_near to keep runtime low.")


def show_image(url_path: str, caption: str):
    full = f"{BACKEND}{url_path}"
    st.image(full, caption=caption, width="stretch")


# ---- UI ----
st.title("AeroPINN-X")
def have_airfoil():
    return st.session_state.airfoil_bytes is not None
if page == "Home":
    render_home()
elif page == "Upload + Preview":
    st.header("Upload + Preview")
    file = st.file_uploader("Upload airfoil .dat", type=["dat"], key="airfoil_uploader")
    st.write("Upload the airfoil file. Then run training to generate point cloud + plots.")
    if file is not None:
        st.success(f"Uploaded: {file.name}")
        st.session_state.airfoil_bytes = file.getvalue()
        st.session_state.airfoil_name = file.name

    if have_airfoil():
        st.write("Current airfoil:", st.session_state.airfoil_name)
elif page == "Train":
    st.header("Train")

    if not have_airfoil():
        st.warning("Upload a .dat airfoil first (Upload + Preview page).")
        st.stop()
    st.write("Airfoil:", st.session_state.airfoil_name)

    if st.button("Run (async)", key="run_async_btn"):
        resp = api_post_run(st.session_state.airfoil_bytes, st.session_state.airfoil_name)
        st.session_state.run_id = resp["run_id"]
        st.success(f"Started run: {st.session_state.run_id}")

    col1, col2 = st.columns(2)
    with col2:
            st.subheader("Current run")
            st.code(st.session_state.get("run_id") or "None")

            if "last_status" not in st.session_state:
                st.session_state.last_status = None

            if st.button("Poll status now", key="poll_btn") and st.session_state.get("run_id"):
                try:
                    status = api_get(f"/runs/{st.session_state.run_id}/status", timeout=10)
                    st.session_state.last_status = status
                except Exception as e:
                    st.session_state.last_status = {"status": "error", "error": str(e)}

            status = st.session_state.get("last_status")

            if not status:
                st.info("No status yet. Click **Poll status now**.")
            else:
                st.write("Run status:", status)
                if status.get("status") == "error":
                    st.error(status.get("error", "Unknown error"))
                elif status.get("status") == "done":
                    st.success("Run completed!")
                elif status.get("status") == "running":
                    st.warning("Running...")
                else:
                    st.info(status.get("status"))



elif page == "Results":
    st.header("Results")

    run_id = st.session_state.run_id
    if not run_id:
        st.info("Run training first.")
    else:
        status = api_get(f"/runs/{run_id}/status")
        st.write("Status:", status["status"])
        if status["status"] != "done":
            st.warning("Run not finished yet. Go to Train page and poll status.")
        else:
            result = api_get(f"/runs/{run_id}/result")
            st.session_state.last_result = result

            arts = result["artifacts"]
            c1, c2 = st.columns(2)
            with c1:
                show_image(arts["speed"], "Speed magnitude √(u²+v²)")
            with c2:
                show_image(arts["pressure"], "Pressure field p")

            st.subheader("Point cloud")
            show_image(arts["point_cloud"], "Mesh-free point cloud")

            st.subheader("PREM summary (numbers)")
            st.json(result.get("prem", {}))

elif page == "PREM Heatmaps":
    st.header("PREM Residual Heatmaps")
    run_id = st.session_state.run_id
    if not run_id:
        st.info("Run training first.")
    else:
        status = api_get(f"/runs/{run_id}/status")
        if status["status"] != "done":
            st.warning("Run not finished yet.")
        else:
            result = api_get(f"/runs/{run_id}/result")
            arts = result["artifacts"]
            c1, c2 = st.columns(2)
            with c1:
                show_image(arts["prem_cont"], "Continuity residual heatmap")
            with c2:
                show_image(arts["prem_sa"], "SA residual heatmap")

elif page == "optimize AOA":
    st.header("Optimize AoA")
    if not have_airfoil():
        st.warning("Upload a .dat airfoil first (Upload + Preview page).")
        st.stop()

    # Optimization controls
    alpha_min = st.sidebar.slider("alpha_min (deg)", -5.0, 15.0, 0.0, 0.5, key="alpha_min")
    alpha_max = st.sidebar.slider("alpha_max (deg)", -5.0, 15.0, 10.0, 0.5, key="alpha_max")
    n_alpha = st.sidebar.selectbox("n_alpha (samples)", [3, 5, 7, 9], index=1, key="n_alpha")

    steps_opt = st.sidebar.slider("steps per alpha", 10, 150, 50, 10, key="steps_opt")

    st.write(f"**Airfoil:** {st.session_state.airfoil_name}")
    st.write(f"**Re:** {Re_phys}")
    st.write(f"**Search:** α in [{alpha_min}, {alpha_max}] with {n_alpha} samples")
    st.write(f"**Per-run:** steps={steps_opt}, N_int={N_int}, N_near={N_near}, lr={lr}")

    if st.button("Run AoA optimization", key="run_opt_btn"):
        resp = api_post_optimize_aoa(
            file_bytes=st.session_state.airfoil_bytes,
            file_name=st.session_state.airfoil_name,
            Re_phys=Re_phys,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_alpha=n_alpha,
            steps=steps_opt,
            N_int=N_int,
            N_near=N_near,
            lr=lr,
        )
        if resp is None:
            st.stop()
        st.session_state.opt_id = resp["opt_id"]
        st.success(f"Started optimization: {st.session_state.opt_id}")

    # optional: polling section similar to Train page (we’ll add in later after backend is ready)

    if st.session_state.opt_id:
        st.subheader("Optimization status")
        st.code(st.session_state.opt_id)

        if st.button("Poll optimization status", key="poll_opt"):
            st.session_state.opt_status = api_get(f"/optimize/{st.session_state.opt_id}/status", timeout=10)

        status = st.session_state.get("opt_status")
        if status:
            st.write(status)

        if status and status.get("status") == "done":
            if st.button("Load optimization result", key="load_opt"):
                st.session_state.opt_result = api_get(f"/optimize/{st.session_state.opt_id}/result", timeout=10)

        result = st.session_state.get("opt_result")
        if result:
            table = result["table"]
            best = result["best"]

            import pandas as pd
            df = pd.DataFrame([{"alpha": r["alpha"], "objective": r["objective"]} for r in table])

            st.subheader("Objective vs AoA")
            st.line_chart(df.set_index("alpha"))

            st.subheader("Best AoA")
            st.success(f"Best α = {best['alpha']} deg | objective = {best['objective']:.4f}")

            # st.subheader("Best artifacts")
            # art = best["artifacts"]
            # st.image(f"{BACKEND}{art['pressure']}")
            # st.image(f"{BACKEND}{art['speed']}")
            # st.image(f"{BACKEND}{art['prem_cont']}")
            # st.image(f"{BACKEND}{art['prem_sa']}")
