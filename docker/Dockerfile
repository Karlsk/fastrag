FROM python:3.13.2-slim

WORKDIR /app

# Disable Python stdout/stderr buffering so logs appear in docker logs immediately
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# --- Dependency cache layer ---
# Copy only pyproject.toml and create a minimal package stub so that
# "uv pip install .[all,ui]" resolves and installs all *dependencies*
# without needing the real source code.  This layer is only rebuilt
# when pyproject.toml changes, not on every code edit.
COPY pyproject.toml .
RUN mkdir -p src/fastrag && echo '__version__ = "0.1.0"' > src/fastrag/__init__.py
RUN uv pip install --system --no-cache ".[all,ui]"

# --- Application code (fast layer, changes often) ---
COPY src/ src/
COPY app/ app/

# Re-install the local package so the real source is picked up
RUN uv pip install --system --no-deps --no-cache .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
