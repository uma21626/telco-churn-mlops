# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
#COPY mlruns/818038774037910443/models/m-3b62448a0966474ba538b825816f7b95/artifacts /app/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
#COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
#COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
#COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl
#COPY mlruns/818038774037910443/7cb3433fe06d4ec098287cae668cdaa3/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/artifacts /app/model

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]