# Bus Service Template

This guide shows how to generate test certificates and run a container built from
the `bus_service` template.

`dtrt bus init service` can optionally set the JetStream stream name, storage
backend and TLS paths when creating a new service. It also allows configuring
the maximum number of messages retained per subject.

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

### CLI Options

`dtrt bus init service` accepts extra flags to customise the generated files:

```bash
dtrt bus init service mysvc \
  --stream-name deepthought_events \
  --tls-cert certs/client-cert.pem \
  --tls-key certs/client-key.pem \
  --tls-ca certs/ca.pem \
  --js-storage file \
  --max-msgs 5000
```

The values supplied are interpolated into `nats.env.example` and the Dockerfile.

## Generated Files

Running `dtrt bus init service <name>` creates a service directory with several
pre-populated files:

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | Minimal image that installs the `deepthought` package and loads TLS certificates. |
| `nats.env.example` | Example environment file containing connection settings for NATS. |
| `service.py` | Skeleton service that forwards messages from `dtr.template.input` to `dtr.template.output`. |
| `publisher.py` | Thin wrapper around `deepthought.eda.Publisher` for publishing events. |
| `subscriber.py` | Example subscriber showing how to apply a rate limit to message handling. |
| `__init__.py` | Empty module marker so Python treats the directory as a package. |

### Environment Variables

The `nats.env.example` file defines credentials and optional mTLS paths used by
both the publisher and subscriber helpers:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `NATS_URL` | URL of the NATS server | `nats://localhost:4222` |
| `NATS_USERNAME` | Username for authentication | `example` |
| `NATS_PASSWORD` | Password for authentication | `secret` |
| `NATS_STREAM` | JetStream stream name | `deepthought_events` |
| `NATS_TLS_CERT` | Path to the client certificate (optional) | *(unset)* |
| `NATS_TLS_KEY` | Path to the client key (optional) | *(unset)* |
| `NATS_TLS_CA` | Path to the CA certificate (optional) | *(unset)* |
| `NATS_JS_STORAGE` | JetStream storage backend | `memory` |
| `NATS_MAX_MSGS` | Max messages per subject | `10000` |

### Rate Limit Decorator

`subscriber.py` defines a `rate_limit` decorator implementing a simple token
bucket algorithm. Apply it to a handler to limit how many messages are processed
per time interval:

```python
@rate_limit(10, 1)  # 10 messages per second
async def _handle(self, msg):
    await msg.ack()
```

The first argument is the bucket capacity and the second is the refill interval
in seconds. When the bucket is empty, the wrapper waits until enough tokens are
available before calling the original handler.
