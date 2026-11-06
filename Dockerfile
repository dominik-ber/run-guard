FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY RunGuard ./RunGuard

COPY raw_data ./raw_data

# Training during the build : produces /app/models/model.pkl
RUN python -m RunGuard.interface.init_model


CMD uvicorn RunGuard.api.fast:app --host 0.0.0.0 --port ${PORT:-8000}
