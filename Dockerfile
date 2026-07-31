FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie des fichers nécessaires
COPY ./app/

EXPOSE 7860

CMD ["uvicorn", "api.api_rag:app", "--host", "0.0.0.0", "--port", "7860"]
