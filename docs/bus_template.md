# Bus Service Template

This guide shows how to generate test certificates and run a container built from
the `bus_service` template.

## Generate Certificates

Use `openssl` to create a simple CA and client certificate pair:

```bash
mkdir certs
openssl req -x509 -newkey rsa:2048 -days 365 -nodes \
    -subj "/CN=test-ca" \
    -keyout certs/ca-key.pem -out certs/ca.pem

openssl req -newkey rsa:2048 -nodes \
    -subj "/CN=test-client" \
    -keyout certs/client-key.pem -out certs/client.csr

openssl x509 -req -days 365 \
    -in certs/client.csr -CA certs/ca.pem -CAkey certs/ca-key.pem \
    -CAcreateserial -out certs/client-cert.pem
```

## Build and Run

Build the image and start the service using the provided `nats.env.example`:

```bash
docker build -t mysvc -f Dockerfile .
docker run --env-file nats.env.example mysvc
```

The container copies certificates from the `certs/` directory and sets
`NATS_TLS_CERT`, `NATS_TLS_KEY` and `NATS_TLS_CA` automatically. Ensure your NATS
server is started with the same certificate files.
