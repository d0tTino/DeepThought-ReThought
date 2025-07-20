FROM golang:1.20
WORKDIR /app
COPY go.mod .
RUN go mod download
COPY . .
RUN go build -o service .
COPY nats.env.example /etc/nats.env
COPY certs/*.pem /certs/
ENV NATS_TLS_CERT=
ENV NATS_TLS_KEY=
ENV NATS_TLS_CA=
ENV NATS_JS_STORAGE=memory
ENV NATS_MAX_MSGS=10000
CMD ["./service"]
