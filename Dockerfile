# SemanticSwap - Memory-Optimized Inference Proxy
# Build:  docker build -t semanticswap .
# Run:    docker run -p 8080:8080 -v semanticswap-data:/data semanticswap
#         (erwartet Ollama auf dem Host unter :11434; siehe config.docker.yaml)
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY config.docker.yaml /app/config.yaml

RUN mkdir -p /data
VOLUME /data
EXPOSE 8080

# Healthcheck gegen den eingebauten Endpoint (Auth-frei)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; \
  sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "semanticswap.main", "/app/config.yaml"]
