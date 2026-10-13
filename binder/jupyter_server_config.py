# Jupyter server config added by this repo (installed to /usr/local/etc/jupyter/,
# loaded alongside the base image's /etc/jupyter/jupyter_server_config.py).
import os

c = get_config()  # noqa: F821

# Serve bambot's static build at <jupyter base url>/bambot/ through
# jupyter-server-proxy. The site uses relative asset paths, so it works under
# whatever base url the server has (e.g. /user/<name>/ on Binderhub).
BAMBOT_DIR = os.environ.get("BAMBOT_DIR", "/opt/bambot")

c.ServerProxy.servers = {
    "bambot": {
        "command": [
            "python3", "-m", "http.server", "{port}",
            "--bind", "127.0.0.1",
            "--directory", BAMBOT_DIR,
        ],
        "timeout": 20,
        "launcher_entry": {
            "title": "Bambot",
            "enabled": True,
        },
    },
}
