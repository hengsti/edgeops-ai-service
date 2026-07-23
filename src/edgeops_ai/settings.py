from pydantic_settings import BaseSettings, SettingsConfigDict

from edgeops_ai.prediction_backends import PredictionBackendName


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    collector_base_url: str = "http://localhost:8095"
    collector_api_key: str = "local-dev-key"
    collector_poll_interval_seconds: float = 10.0
    collector_timeout_seconds: float = 3.0
    collector_polling_enabled: bool = True
    collector_device_ids: str = ""

    prediction_backend: PredictionBackendName = PredictionBackendName.RULES

    @property
    def device_ids(self) -> tuple[str, ...]:
        return tuple(
            device_id.strip()
            for device_id in self.collector_device_ids.split(",")
            if device_id.strip()
        )
