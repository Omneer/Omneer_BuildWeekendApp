
FROM python:3.10 

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 8080 8000
# Heroku uses PORT, Azure App Services uses WEBSITES_PORT, Fly.io uses 8080 by default
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]