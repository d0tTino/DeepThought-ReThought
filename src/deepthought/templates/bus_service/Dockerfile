FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ../../..
# Example credentials file for NATS connection
COPY nats.env.example /etc/nats.env
# Copy TLS certificates if present
COPY certs/*.pem /certs/
# Configure mTLS inside the container
ENV NATS_TLS_CERT=/certs/client-cert.pem
ENV NATS_TLS_KEY=/certs/client-key.pem
ENV NATS_TLS_CA=/certs/ca.pem
CMD ["python", "-m", "template_service"]
