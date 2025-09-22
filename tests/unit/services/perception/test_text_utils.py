from deepthought.services.perception.text_utils import hop_aligned_tokens, scrub_tokens


def test_hop_aligned_tokens_redacts_pii():
    tokens = hop_aligned_tokens("Contact me at 555-123-4567", 0.1)
    assert tokens == [
        ("Contact", 0.0, 0.1),
        ("me", 0.1, 0.2),
        ("at", 0.2, 0.30000000000000004),
        ("[REDACTED]", 0.30000000000000004, 0.4),
    ]


def test_scrub_tokens_casts_and_redacts():
    raw_tokens = [["foo@example.com", "0", "0.5"], ("hello", 0.5, 1)]
    sanitized = scrub_tokens(raw_tokens)
    assert sanitized == [
        ("[REDACTED]", 0.0, 0.5),
        ("hello", 0.5, 1.0),
    ]
