import re
from pathlib import Path
from typing import Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgeops_ai.prediction import PredictionBackendName


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    collector_base_url: str = "http://localhost:8095"
    collector_api_key: str = "local-dev-key"
    collector_poll_interval_seconds: float = 10.0
    collector_timeout_seconds: float = 3.0
    collector_polling_enabled: bool = True
    collector_device_ids: str = ""

    prediction_backend: PredictionBackendName = PredictionBackendName.RULES
    model_artifact_path: Path = Path("artifacts/model.onnx")
    model_artifact_sha256: str | None = None

    advisor_influx_url: HttpUrl | None = None
    advisor_influx_org: str = Field(default="smarthome", min_length=1)
    advisor_influx_token: SecretStr | None = None
    advisor_refresh_seconds: float = Field(default=300, ge=60, le=3600)

    @model_validator(mode="after")
    def validate_advisor(self) -> Self:
        token = self.advisor_influx_token
        if bool(self.advisor_influx_url) != bool(token and token.get_secret_value().strip()):
            raise ValueError(
                "Set both ADVISOR_INFLUX_URL and ADVISOR_INFLUX_TOKEN to enable Advisor."
            )
        return self

    @model_validator(mode="after")
    def validate_model_artifact(self) -> Self:
        digest = (self.model_artifact_sha256 or "").strip().lower()
        if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("MODEL_ARTIFACT_SHA256 must contain exactly 64 hexadecimal digits.")
        if self.prediction_backend is PredictionBackendName.ML and not digest:
            raise ValueError("MODEL_ARTIFACT_SHA256 is required for the ml backend.")
        self.model_artifact_sha256 = digest or None
        return self

    @property
    def device_ids(self) -> tuple[str, ...]:
        return tuple(
            device_id.strip()
            for device_id in self.collector_device_ids.split(",")
            if device_id.strip()
        )
