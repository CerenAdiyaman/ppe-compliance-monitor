"""Application settings loaded from environment variables.

Every setting is an ``HG_``-prefixed environment variable or ``.env`` entry;
see ``.env.example``. Import the module-level ``settings`` singleton rather
than instantiating ``Settings`` again, so the whole process shares one
configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HG_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # CLI flags are applied by assigning to this singleton, so the same
        # constraints that guard `.env` must also guard those assignments.
        validate_assignment=True,
    )

    # --- Ingestion ---
    # Webcam index ("0"), a file path, or an rtsp:// URL. Kept as a string
    # because OpenCV needs an int for cameras and a str for everything else;
    # VideoSource does that conversion.
    video_source: str = "0"

    # Which OpenCV capture backend to use for webcams. "auto" tries OpenCV's
    # own preference and falls back to DirectShow, which on Windows frequently
    # succeeds where Media Foundation opens the device but never delivers a
    # frame. Set this explicitly to skip the probe and its startup cost.
    capture_backend: Literal["auto", "any", "dshow", "msmf"] = "auto"

    # --- Detection ---
    model_path: Path = BASE_DIR / "models" / "best.pt"
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # "cpu", "cuda", or a specific GPU such as "cuda:0".
    device: str = "cpu"

    # Class ids from the trained model. `person` is excluded: it was
    # undertrained (2% of instances) and the rule logic does not need it.
    class_head: int = 0
    class_helmet: int = 1

    # --- Violation rules ---
    # Consecutive frames a violation must persist before it is logged.
    confirmation_frames: int = Field(default=5, ge=1)
    # Minimum gap between alerts of the same type, per camera.
    cooldown_seconds: int = Field(default=60, ge=0)

    # --- Privacy ---
    anonymize_faces: bool = True

    # --- Storage ---
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'violations.db'}"
    violations_dir: Path = BASE_DIR / "data" / "violations"

    # --- Notification ---
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    alert_recipient: str = ""
    notifications_enabled: bool = False

    @field_validator("device")
    @classmethod
    def _known_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "cpu" and not value.startswith("cuda"):
            raise ValueError("device must be 'cpu', 'cuda' or 'cuda:<index>'")
        return value

    @model_validator(mode="after")
    def _credentials_present(self) -> "Settings":
        if self.notifications_enabled:
            missing = [
                name
                for name in ("smtp_user", "smtp_password", "alert_recipient")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "notifications_enabled requires "
                    + ", ".join(f"HG_{n.upper()}" for n in missing)
                )
        return self


settings = Settings()


# Camera inventory. Kept out of Settings because it is a structure, not a
# scalar knob: a single monitored site rarely changes it between runs, and a
# multi-camera deployment is better served editing this table than encoding
# nested JSON in an environment variable.
CAMERAS: dict[str, dict[str, str]] = {
    "CAM_01": {
        "source": settings.video_source,  # webcam index, file path or rtsp:// URL
        "location": "Assembly Line A",
    },
}
