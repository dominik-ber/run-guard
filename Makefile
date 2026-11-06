# Build and run localy in a container
docker-build:
	docker build --tag=runguard:dev .


docker-build-prod:
	docker build --tag=runguard:prod .


docker-run:
	docker run -it -e PORT=8000 -p 8080:8000 runguard:dev


run-docker:
	docker run -it --env-file .env -p 8080:8000 runguard:dev

docker-build-gcp:
	docker build \
		--platform linux/amd64 \
		-t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$IMAGE:prod \
  .




# build for GCP
build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.
