FROM python:3.11-slim

WORKDIR /app

ENV DEPLOYMENT_MODE=lightweight
ENV PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=10000"]