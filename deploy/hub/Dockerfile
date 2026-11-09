# MLPilot Hub server image. Build from the repository root:
#   docker build -f deploy/hub/Dockerfile -t mlpilot-hub .
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /src
COPY . .
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[server]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AC_OFFLINE=1 \
    AC_HUB_ARTIFACTS=/data/artifacts AC_HUB_DATABASE_URL=sqlite:////data/hub.db
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels \
    && useradd --create-home --uid 10001 hub && mkdir -p /data && chown hub:hub /data
USER hub
WORKDIR /home/hub
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status == 200 else 1)"
CMD ["mlpilot", "hub", "serve", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
