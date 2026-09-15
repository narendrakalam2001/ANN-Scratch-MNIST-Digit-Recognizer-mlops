# ============================================================
# Dockerfile — ANN From Scratch: MNIST Digit Recognizer API
# ============================================================
FROM python:3.12-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY src/ src/
COPY serving/ serving/
COPY services/ services/
COPY ann_models/ ann_models/
COPY scripts/run_api.py scripts/run_api.py

ENV PORT=8000
EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
