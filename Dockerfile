FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN groupadd --gid 10001 apexflow && useradd --uid 10001 --gid apexflow --create-home apexflow

FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["pytest", "-q", "-o", "cache_dir=/tmp/apexflow-pytest-cache"]
USER apexflow

FROM base AS runtime
USER apexflow
CMD ["python", "-m", "apexflow", "--data", "/app/data", "--output", "/app/output"]
