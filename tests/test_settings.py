import pytest
from pydantic import ValidationError

from edgeops_ai.settings import Settings


def test_ml_settings_require_artifact_sha256() -> None:
    with pytest.raises(ValidationError, match="MODEL_ARTIFACT_SHA256 is required"):
        Settings(_env_file=None, prediction_backend="ml")


@pytest.mark.parametrize("digest", ["short", "g" * 64, "0" * 63, "0" * 65])
def test_settings_reject_invalid_artifact_digest(digest: str) -> None:
    with pytest.raises(ValidationError, match="64 hexadecimal"):
        Settings(_env_file=None, model_artifact_sha256=digest)


def test_settings_normalize_artifact_digest() -> None:
    settings = Settings(
        _env_file=None, prediction_backend="ml", model_artifact_sha256="  " + "A" * 64 + "  "
    )
    assert settings.model_artifact_sha256 == "a" * 64


def test_rules_settings_do_not_require_an_artifact_digest() -> None:
    assert Settings(_env_file=None, prediction_backend="rules").model_artifact_sha256 is None


@pytest.mark.parametrize("digest", ["", "  "])
def test_blank_digest_keeps_documented_rules_configuration_working(digest: str) -> None:
    assert (
        Settings(
            _env_file=None, prediction_backend="rules", model_artifact_sha256=digest
        ).model_artifact_sha256
        is None
    )


@pytest.mark.parametrize("digest", ["", "  "])
def test_ml_settings_reject_blank_digest(digest: str) -> None:
    with pytest.raises(ValidationError, match="MODEL_ARTIFACT_SHA256 is required"):
        Settings(_env_file=None, prediction_backend="ml", model_artifact_sha256=digest)
