# Run Guard

Run Guard is a small FastAPI service scaffold for exposing a prediction function over HTTP. The current prediction implementation is a placeholder and returns `"OK"`.

## Requirements

- Python 3.10
- FastAPI and Uvicorn

`requirements.txt` is currently empty, so install the runtime packages directly for local development:

```bash
python -m pip install fastapi uvicorn
```

## Run Locally

From the repository root, start the API with:

```bash
uvicorn package_folder.api_file:app --reload --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

## API

### `GET /`

Returns a greeting:

```json
{"greeting":"Hello"}
```

### `GET /predict`

Calls the prediction function and returns its result:

```json
{"prediction":"OK"}
```

Example request:

```bash
curl http://localhost:8000/predict
```

## Docker

The Dockerfile starts Uvicorn on the port specified by the `PORT` environment variable. The current `requirements.txt` is empty, however, so the image does not install FastAPI or Uvicorn and will not start until those packages are added to that file.

Once the runtime dependencies are listed in `requirements.txt`, build and run the image with:

```bash
docker build --tag run-guard:dev .
docker run --rm -e PORT=8000 -p 8080:8000 run-guard:dev
```

The service will then be available at `http://localhost:8080`.

## Google Cloud Run

The Makefile includes targets for building and deploying a container to Cloud Run. Configure `IMAGE`, `GCP_PROJECT`, `GCP_REGION`, `ARTIFACTSREPO`, and `MEMORY` in your shell, and authenticate with the Google Cloud CLI before using these targets. For example:

```bash
make build_for_production IMAGE=run-guard GCP_PROJECT=my-project GCP_REGION=europe-west1 ARTIFACTSREPO=my-repository
make push_image_production IMAGE=run-guard GCP_PROJECT=my-project GCP_REGION=europe-west1 ARTIFACTSREPO=my-repository
make deploy_to_cloud_run IMAGE=run-guard GCP_PROJECT=my-project GCP_REGION=europe-west1 ARTIFACTSREPO=my-repository MEMORY=512Mi
```

Cloud Run deployment also depends on a working container image, so add the runtime dependencies to `requirements.txt` first.
