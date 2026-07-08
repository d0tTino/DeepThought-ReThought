from pathlib import Path

from deepthought.eda.contracts import (
    CanonicalSubjects,
    translate_egress_subject,
    translate_ingress_subject,
)
from deepthought.eda.events import EventSubjects
from deepthought.eda.subject_validator import (
    is_canonical_subject,
    validate_service_bindings,
    validate_service_modules,
)


def test_event_subjects_are_canonical_for_internal_services():
    noncanonical = {
        name: value
        for name, value in vars(EventSubjects).items()
        if name.isupper() and isinstance(value, str) and not is_canonical_subject(value)
    }
    assert noncanonical == {}


def test_production_services_use_canonical_subjects():
    violations = validate_service_modules(
        [
            Path("src/deepthought/services"),
            Path("src/deepthought/modules"),
        ]
    )
    assert violations == []


def test_orchestrator_bindings_declare_canonical_subjects():
    assert validate_service_bindings(Path("examples/orchestrator.yml")) == []


def test_legacy_translation_is_boundary_only():
    assert translate_ingress_subject("dtr.input.received") == CanonicalSubjects.INPUT_RECEIVED
    assert translate_egress_subject(CanonicalSubjects.INPUT_RECEIVED) == CanonicalSubjects.INPUT_RECEIVED
    assert translate_egress_subject(CanonicalSubjects.INPUT_RECEIVED, legacy=True) == "dtr.input.received"
