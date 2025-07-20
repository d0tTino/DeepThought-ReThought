package main

import (
    "log"
    "github.com/nats-io/nats.go"
)

type TemplateService struct {
    js nats.JetStreamContext
}

func NewTemplateService(js nats.JetStreamContext) *TemplateService {
    return &TemplateService{js: js}
}

func (s *TemplateService) Start() error {
    _, err := s.js.Subscribe("dtr.template.input", func(m *nats.Msg) {
        s.js.Publish("dtr.template.output", m.Data)
        m.Ack()
    }, nats.Durable("template_service_listener"))
    return err
}

func main() {
    nc, err := nats.Connect(nats.DefaultURL)
    if err != nil {
        log.Fatal(err)
    }
    js, _ := nc.JetStream()

    svc := NewTemplateService(js)
    if err := svc.Start(); err != nil {
        log.Fatal(err)
    }

    select {}
}
