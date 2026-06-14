FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    JUPYTER_CONFIG_DIR=/tmp/jupyter-config \
    JUPYTER_DATA_DIR=/tmp/jupyter-data \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY retail_pricing/ retail_pricing/
COPY Middle_MLE_HW.ipynb README.md retail_mle_task_data.parquet ./

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p artifacts \
    /tmp/matplotlib \
    /tmp/jupyter-config \
    /tmp/jupyter-data \
    /tmp/jupyter-runtime \
    && chown -R appuser:appuser /workspace \
    /tmp/matplotlib \
    /tmp/jupyter-config \
    /tmp/jupyter-data \
    /tmp/jupyter-runtime

USER appuser

EXPOSE 8888

CMD ["jupyter", "lab", "Middle_MLE_HW.ipynb", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token="]
