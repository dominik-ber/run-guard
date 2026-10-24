# 1. Base Image: Python 3.10 slim
FROM python:3.10.6-slim

# Set working directory inside the container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Package Folder (application code)
COPY package_folder/ ./package_folder/

#COPY models/ ./models/ TODO

# 6. Specify Run Command using Uvicorn for FastAPI
CMD uvicorn package_folder.api_file:app --host 0.0.0.0 --port ${PORT}
