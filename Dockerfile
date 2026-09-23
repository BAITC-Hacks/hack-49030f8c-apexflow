FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements*.txt .
RUN pip install --no-cache-dir -c requirements-lock.txt -r requirements.txt && pip check
COPY . .
RUN groupadd --gid 10001 apexflow && useradd --uid 10001 --gid apexflow --create-home apexflow
ARG VCS_REF=unknown
ENV APEXFLOW_REVISION=$VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF

FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -c requirements-lock.txt -r requirements-dev.txt && pip check
CMD ["pytest", "-q", "-o", "cache_dir=/tmp/apexflow-pytest-cache"]
USER apexflow

FROM base AS runtime
USER apexflow
CMD ["python", "-m", "apexflow", "--data", "/app/data", "--output", "/app/output"]
