import { connect, StringCodec } from "nats";

export class TemplateService {
  constructor(private server: string) {}

  async start() {
    const nc = await connect({ servers: this.server });
    const js = nc.jetstream();
    const sc = StringCodec();
    const sub = await js.subscribe("dtr.template.input", { durable: "template_service_listener" });
    for await (const m of sub) {
      await js.publish("dtr.template.output", m.data);
      m.ack();
    }
  }
}

(async () => {
  const svc = new TemplateService("nats://localhost:4222");
  await svc.start();
})();
