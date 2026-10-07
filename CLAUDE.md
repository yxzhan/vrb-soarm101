# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binder template for running robotics research Jupyter Notebooks on Binderhub. Built on a ROS 2 (Jazzy) + Python 3.12 Docker image (`intel4coro/jupyter-ros2:jazzy-py3.12`) with JupyterLab and code-server (VS Code in browser).

## Build & Run

```bash
# Build and run locally (from repo root)
docker compose -f ./binder/docker-compose.yml up --build

# Stop and remove container
docker compose -f ./binder/docker-compose.yml down

# Install Python dependencies (inside container)
pip install -r requirements.txt
```

Access JupyterLab at http://localhost:8888/ after starting.

Ports exposed by docker-compose: `8888` (JupyterLab), `9090` (rosbridge WebSocket).

## Architecture

- **`binder/`** — Docker environment: `Dockerfile` (base image + AI coding tools + rosbridge + dependencies), `docker-compose.yml` (local dev), `entrypoint.sh` (sources ROS 2 setup at container start via `${ROS_PATH}/setup.bash`)
- **`notebooks/`** — Jupyter notebooks and Python utilities. `utils.py` provides `display_desktop()` for embedding a remote desktop viewer in JupyterLab via Sidecar widget
- **`bambot/`** — Git submodule (`yxzhan/bambot`). Next.js web app for visualising and controlling the SO-ARM101, including webcam hand-tracking and WebXR (VR controller) control. Clone with `--recurse-submodules`; update with `git submodule update --remote bambot`
- **`mediapipe-samples-web/`** — Git submodule (`google-ai-edge/mediapipe-samples-web`). Reference implementations for the MediaPipe web tasks; `src/tasks/hand-landmarker.ts` is the sample the hand control in `bambot/` is based on
- **`docs/`** — Supplementary documentation (e.g. `rosbridge.md`)
- **`requirements.txt`** — Python deps installed at Docker build time

### Dependency layers in the Dockerfile (ordered for cache efficiency)

1. Heavy, rarely-changing tools: ollama, opencode, claude CLI, code-server extensions
2. `ros-jazzy-rosbridge-suite` apt package (+ `rm -rf /var/lib/apt/lists/*` in same `RUN`)
3. `COPY requirements.txt` + `pip install` — invalidated only when deps change
4. `COPY . ${REPO_DIR}/` — invalidated on any file change

New apt packages go in step 2 (grouped with rosbridge). New Python packages go in `requirements.txt`.

### ROS 2 / rosbridge

- rosbridge WebSocket server runs on port `9090`, started with:
  ```bash
  source /opt/ros/jazzy/setup.bash
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml
  ```
- The conda Python env is separate from system Python. rosbridge's Python deps (`pymongo`, `cbor2`, `ujson`, `tornado`) must be in `requirements.txt` to be available.
- In notebooks, always guard `rclpy.init()` with `if not rclpy.ok()` to avoid errors on re-run.
- Teleop topics (both `sensor_msgs/JointState`, radians): the commander (a bambot page in
  the Commander role) publishes `/joint_commands`; the follower (a bambot page in the
  Follower role, or the MuJoCo sim in `notebooks/soarm101_mujoco/`) follows them and is
  the only publisher of `/joint_states`. Run the sim with `--no-state` when a real
  follower already publishes `/joint_states`.

## Code Style

- PEP 8, 4-space indentation, 100-char line limit
- Type hints on function signatures
- Google/NumPy-style docstrings
- Import order: stdlib, third-party, local (separated by blank lines)
- Naming: `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)

## No Test Framework

No test framework is currently configured. If adding tests, use pytest.
