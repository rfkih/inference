FROM python:3.12-slim

# LightGBM requires OpenMP at runtime (libgomp). No HTTP client tools needed:
# the compose healthcheck uses python's stdlib urllib (see docker-compose.vps.yml).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

CMD ["python", "-m", "blackheart_inference"]
