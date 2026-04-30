"""
modal_app.py — Deploy the Flask backend to Modal as a persistent web endpoint.

Deploy:   cd backend && modal deploy modal_app.py
Dev:      cd backend && modal serve modal_app.py
Logs:     modal app logs stylevector-backend
"""

import modal

# ── Image definition ──────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask==3.0.3",
        "flask-cors==4.0.1",
        "pymongo==4.8.0",
        "python-dotenv==1.0.1",
        "requests==2.32.3",
        "gunicorn==22.0.0",
        "asgiref==3.8.1",      # wraps Flask (WSGI) as ASGI for Modal
    )
    .add_local_dir(".", remote_path="/app")   # copies your whole backend/ folder
)

# ── Secrets — env vars injected at runtime ────────────────────────────────
secrets = [modal.Secret.from_name("stylevector-secrets")]

# ── App ───────────────────────────────────────────────────────────────────
app = modal.App(
    name="stylevector-backend",
    image=image,
    secrets=secrets,
)


@app.function(
    # Scale to zero when idle (saves money)
    min_containers=0,
    max_containers=5,
    # Keep container warm for 5 min after last request
    scaledown_window=300,
    # CPU container — no GPU needed
    cpu=1.0,
    memory=512,
)
@modal.asgi_app()
def flask_app():
    """Entry point — Modal calls this to get the WSGI/ASGI app."""
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from asgiref.wsgi import WsgiToAsgi
    from app import create_app

    flask_application = create_app()
    return WsgiToAsgi(flask_application)
