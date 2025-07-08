FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ../../..
# Example credentials file for NATS connection
COPY nats.env.example /etc/nats.env
# Uncomment and update these lines to enable mTLS inside the container
# ENV NATS_TLS_CERT=/certs/client-cert.pem
# ENV NATS_TLS_KEY=/certs/client-key.pem
# ENV NATS_TLS_CA=/certs/ca.pem
CMD ["python", "-m", "template_service"]
