import deepthought.cli.template_helpers as th


def test_apply_bus_substitutions_escape() -> None:
    template = "\n".join(
        [
            "NATS_TLS_CERT=",
            "NATS_TLS_KEY=",
            "ENV NATS_TLS_CERT=",
            "ENV NATS_TLS_KEY=",
        ]
    )
    result = th.apply_bus_substitutions(
        template,
        tls_cert="cert path.pem",
        tls_key='key"path.pem',
    )
    assert 'NATS_TLS_CERT="cert path.pem"' in result
    assert 'NATS_TLS_KEY="key\\"path.pem"' in result
    assert 'ENV NATS_TLS_CERT="cert path.pem"' in result
    assert 'ENV NATS_TLS_KEY="key\\"path.pem"' in result
