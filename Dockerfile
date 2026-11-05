# AlphaForge API — backend image (FastAPI + real engine on uvicorn:8003)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only what the API needs for fast, reliable builds. The full
# requirements.txt pulls heavy ML deps (torch, xgboost) that the HTTP API
# does not import, so we install a lean runtime set here. yfinance is required
# because `import alphaforge` eagerly imports DataFetcher (api never calls it,
# but the package __init__ chain references it). Each floor is the same tested
# floor pyproject.toml declares (tests/test_packaging.py enforces it).
RUN pip install \
    "fastapi>=0.136.3" \
    "uvicorn>=0.49.0" \
    "pandas>=3.0.3" \
    "numpy>=2.4.6" \
    "scipy>=1.17.1" \
    "yfinance>=1.4.1" \
    "requests>=2.34.2"

# Copy the engine package (alphaforge/api/ ships inside it).
COPY alphaforge/ ./alphaforge/

# Non-root runtime — the API needs no write access or privileges.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

# alphaforge.api.app:main() runs uvicorn on 0.0.0.0:8003
CMD ["python", "-m", "alphaforge.api.app"]
