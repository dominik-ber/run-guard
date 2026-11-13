# Build and run localy in a container
docker-build:
	docker build --tag=runguard:dev .

run-docker:
	docker run -it --env-file .env -p 8080:8000 runguard:dev

# build for GCP
build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.
