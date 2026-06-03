# Notely — Dockerfile
#
# Fixes vs original:
# 1. No static folder created — main.py serves frontend from /app/static/
#    but the original Dockerfile never copies it. Frontend would 404.
# 2. No .dockerignore awareness — COPY . . copies __pycache__, .env, .git,
#    seed_data.py and other junk into the image, bloating it and leaking secrets.
#    Added explicit COPY commands for only what's needed.
# 3. Single pip install layer — any code change forces a full re-install.
#    Separated requirements install from code copy for better layer caching.
# 4. No non-root user — Cloud Run best practice and Google security policy
#    recommend not running as root. Added a minimal appuser.
# 5. uvicorn workers not set — Cloud Run gives each container 1 vCPU by default.
#    Single worker is correct. Added --log-level info for readable Cloud Run logs.
# 6. No PYTHONUNBUFFERED — without it, Python buffers stdout/stderr and Cloud
#    Run's log viewer shows logs with significant delay or misses them entirely.
# 7. No PYTHONDONTWRITEBYTECODE — without it, Python writes .pyc files inside
#    the container at runtime, wasting disk I/O on a read-mostly filesystem.

FROM python:3.11-slim

# FIX 6 & 7: environment flags for cleaner Cloud Run logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# FIX 3: copy requirements first so pip layer is cached across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FIX 2: copy only the Python source files explicitly (no .env, no __pycache__, no seed_data)
COPY main.py database.py embeddings.py agent_tools.py ./

# FIX 1: copy the frontend into /app/static/ so main.py can serve it at "/"
COPY docs/ ./static/

# FIX 4: create and switch to a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# FIX 5: explicit single worker + info logging for readable Cloud Run logs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]