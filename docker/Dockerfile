FROM python:3.10-slim

WORKDIR /app

COPY docker/app.py /app/app.py
COPY model.joblib /app/model.joblib
COPY feature_columns.txt /app/feature_columns.txt
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
