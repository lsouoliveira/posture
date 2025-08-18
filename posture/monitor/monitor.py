from abc import ABC
from typing import Any, Optional
import time

from posture.detector.detector import Detector, Posture
from posture.detector.inference_model import InferenceModel
from posture.camera import Camera, CouldNotOpenDeviceFound
from posture.logging import get_logger

logger = get_logger(__name__)

RETRY_INTERVAL = 5


def backoff_interval(retries: int) -> float:
    return min(RETRY_INTERVAL * (2**retries), 600)


class Observable(ABC):
    def __init__(self):
        self._callbacks = []

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def unsubscribe(self, callback):
        self._callbacks.remove(callback)

    def notify_observers(self, data: Any):
        for callback in self._callbacks:
            callback(data)


class PostureEvent:
    def __init__(self, posture: Posture):
        self.posture = posture


class Error(Exception):
    pass


class NoCameraFound(Error):
    def __init__(self):
        super().__init__("No camera found or could not open the video device")


class CaptureError(Error):
    def __init__(self):
        super().__init__("Could not capture frame from the video device")


class NotRunningError(Error):
    def __init__(self):
        super().__init__("Posture monitor is not running")


class AlreadyRunningError(Error):
    def __init__(self):
        super().__init__("Posture monitor is already running")


def with_retry(func):
    def wrapper(*args, **kwargs):
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except (NoCameraFound, CaptureError) as e:
                retries += 1
                retry_interval = backoff_interval(retries)

                logger.debug(
                    f"Error occurred: {e}, retrying in {retry_interval} seconds..."
                )

                time.sleep(retry_interval)

    return wrapper


class PostureMonitor(Observable):
    def __init__(
        self,
        inference_model: InferenceModel,
        monitor_interval: float,
        camera: Optional[Camera] = None,
    ):
        super().__init__()

        self.camera = camera or Camera()
        self.monitor_interval = monitor_interval
        self.detector = Detector(inference_model)
        self._is_running = False

    def start(self):
        if self._is_running:
            raise AlreadyRunningError()

        self._is_running = True

        self._mainloop()

    def stop(self):
        if not self._is_running:
            raise NotRunningError()

        self._is_running = False
        self._close_camera()

    @with_retry
    def _mainloop(self):
        while self._is_running:
            posture = self._detect_posture()

            if posture is not None:
                self.notify_observers(PostureEvent(posture))

            time.sleep(self.monitor_interval)

    def _detect_posture(self):
        try:
            self._open_camera()

            return self.detector.detect(self._capture_frame()) or Posture.unknown()
        finally:
            self._close_camera()

    def _open_camera(self):
        try:
            self.camera.open()
        except CouldNotOpenDeviceFound:
            raise NoCameraFound()

    def _close_camera(self):
        if self.camera:
            self.camera.close()

    def _capture_frame(self):
        return self.camera.capture()
