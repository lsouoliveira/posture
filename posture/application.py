import sys
from typing import Any
import json

from posture.detector.inference_model import (
    InferenceModel,
    Error as InferenceModelError,
)
from posture.monitor.monitor import PostureEvent, PostureMonitor
from posture.config import settings
from posture.logging import get_logger

logger = get_logger(__name__)


class Application:
    def __init__(self):
        self._inference_model = None
        self._posture_monitor = None

    def init(self):
        logger.debug("Initializing Posture Monitor Application")
        logger.debug(f"Using model path: {settings.MODEL_PATH}")

        self._inference_model = InferenceModel(settings.MODEL_PATH)

        try:
            self._inference_model.load()
        except InferenceModelError as e:
            logger.error(f"Failed to load inference model: {e}")
            sys.exit(1)

        self._posture_monitor = PostureMonitor(
            self._inference_model, settings.MONITOR_INTERVAL
        )
        self._posture_monitor.subscribe(self._handle_posture_event)

    def run(self):
        if self._posture_monitor is None:
            logger.error("Posture Monitor was not initialized")
            sys.exit(1)

        try:
            self._posture_monitor.start()
        except Exception as e:
            logger.error(f"An error occurred while running the posture monitor: {e}")
            sys.exit(1)

    def _handle_posture_event(self, event: Any):
        if not isinstance(event, PostureEvent):
            return

        print(json.dumps(self._format_event(event)))

    def _format_event(self, event: PostureEvent) -> str:
        payload = {
            "posture": {
                "type": event.posture.posture_type.value,
                "confidence": event.posture.confidence,
                "bounding_box": list(event.posture.bounding_box),
            }
        }

        return str(payload)
