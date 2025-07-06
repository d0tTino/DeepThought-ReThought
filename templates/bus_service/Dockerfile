FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ../../..
# Example credentials file for NATS connection
COPY nats.env.example /etc/nats.env
CMD ["python", "-m", "template_service"]
