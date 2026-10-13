# Build bambot's static site in its own stage, so it is rebuilt only when
# bambot changes and its node_modules stay out of the final image.
FROM node:22-slim AS bambot
RUN npm i -g pnpm@9
WORKDIR /bambot
COPY bambot/package.json bambot/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY bambot/ ./
RUN pnpm build

# Dockerfile of the base image: https://github.com/IntEL4CoRo/jupyter-ros2/blob/main/Dockerfile
FROM intel4coro/jupyter-ros2:jazzy-py3.12

USER root
#============= Install extra software packages =============#
# RUN apt update && apt install -y curl

# Install vibe coding tools
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN curl -fsSL https://opencode.ai/install | bash
RUN curl -fsSL https://claude.ai/install.sh | bash
RUN code-server --force --install-extension anthropic.claude-code
ENV PATH=$HOME/.local/bin:$PATH

# Upgrade code-server
RUN mamba update -y conda-forge::code-server
# Install Github Copilot extension
ARG COPILOT_PATH=GitHub.copilot-chat.0.40.1.universal.vsix
RUN wget https://github.com/microsoft/vscode-copilot-chat/releases/download/v0.40.1/${COPILOT_PATH} && \
    code-server --install-extension ./${COPILOT_PATH} && \
    rm ./${COPILOT_PATH}

# Install ROS 2 bridge and Python packages
RUN apt-get update && apt-get install -y ros-jazzy-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN npm i -g pnpm

# bambot, served by Jupyter at <base url>/bambot/ (binder/jupyter_server_config.py)
COPY --from=bambot /bambot/out /opt/bambot
COPY binder/jupyter_server_config.py /usr/local/etc/jupyter/jupyter_server_config.py
#===========================================================#

# Copy content to the image
ENV REPO_DIR=/home/repo
RUN mkdir -p ${REPO_DIR}
COPY . ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}

# Set JupyterLab default directory
WORKDIR ${REPO_DIR}

# Set VSCODE default directory
ENV CODE_WORKING_DIRECTORY=${REPO_DIR}

# The entrypoint of the docker image
COPY binder/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
