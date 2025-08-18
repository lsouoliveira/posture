from enum import Enum
from typing import Optional, cast

from .inference_model import InferenceModel


class PostureType(Enum):
    GOOD = "good"
    BAD = "bad"
    UNKNOWN = "unknown"

    @staticmethod
    def from_label(label: int):
        if label == 0:
            return PostureType.GOOD

        if label == 1:
            return PostureType.BAD

        raise Exception(f"Invalid label value: {label}")


class Posture:
    def __init__(
        self,
        posture_type: PostureType,
        confidence: float,
        bounding_box: tuple[int, int, int, int],
    ):
        self.posture_type = posture_type
        self.confidence = confidence
        self.bounding_box = bounding_box

    def __str__(self):
        return f"Posture(type={self.posture_type}, confidence={self.confidence}, bounding_box={self.bounding_box})"

    @staticmethod
    def unknown():
        return Posture(PostureType.UNKNOWN, 0.0, (0, 0, 0, 0))


class Detector:
    def __init__(self, model: InferenceModel):
        self.model = model

    def detect(self, image) -> Optional[Posture]:
        results = self._predict(image)

        if not results:
            return None

        (x1, y1, x2, y2, label, confidence) = results

        if not x1 or not y1 or not x2 or not y2 or label is None or confidence is None:
            return None

        return Posture(PostureType.from_label(label), confidence, (x1, y1, x2, y2))

    def _predict(
        self, image
    ) -> Optional[
        tuple[int | None, int | None, int | None, int | None, int | None, float | None]
    ]:
        return InferenceModel.get_results(self.model.predict(image))
