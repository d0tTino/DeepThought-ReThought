FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ../../..
# Example credentials file for NATS connection
COPY nats.env.example /etc/nats.env
# Copy TLS certificates if present
COPY certs/*.pem /certs/
# Configure mTLS inside the container if certificates are provided
ENV NATS_TLS_CERT=
ENV NATS_TLS_KEY=
ENV NATS_TLS_CA=
CMD ["python", "-m", "template_service"]
