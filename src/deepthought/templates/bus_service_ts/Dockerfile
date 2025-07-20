FROM node:18
WORKDIR /app
COPY package.json tsconfig.json ./
RUN npm install
COPY src ./src
COPY nats.env.example /etc/nats.env
COPY certs/*.pem /certs/
ENV NATS_TLS_CERT=
ENV NATS_TLS_KEY=
ENV NATS_TLS_CA=
ENV NATS_JS_STORAGE=memory
ENV NATS_MAX_MSGS=10000
CMD ["npm", "run", "start"]
