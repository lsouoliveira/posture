from pathlib import Path

import torch
import yolov5
from yolov5.models.yolo import DetectionModel

from posture.config import settings


def get_highest_memory_device():
    device_memory = {}

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_memory[i] = props.total_memory

    return max(device_memory, key=device_memory.get)


class Error(Exception):
    pass


class ModelNotLoadedError(Error):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"ModelNotLoadedError: {self.message}"


class InferenceModel:
    def __init__(self, path: str):
        self.model_path = Path(path)
        self.model = None

    def load(self):
        try:
            self.model = yolov5.load(
                str(self.model_path),
                device=self.get_device(),
            )
        except Exception as e:
            raise Error("Failed to load model on CPU: {}".format(str(e))) from e

        self.initialize_model(self.model)

        return self

    def initialize_model(self, model):
        model.conf = settings.MODEL_CONFIDENCE_THRESHOLD  # NMS confidence threshold
        model.iou = settings.MODEL_IOU_THRESHOLD  # NMS IoU threshold
        model.classes = settings.MODEL_CLASSES  # Only show these classes
        model.agnostic = settings.MODEL_AGNOSTIC  # NMS class-agnostic
        model.multi_label = settings.MODEL_MULTI_LABEL  # NMS multiple labels per box
        model.max_det = settings.MODEL_MAX_DET  # maximum number of detections per image
        model.amp = settings.MODEL_AMP  # Automatic Mixed Precision (AMP) inference

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:{}".format(get_highest_memory_device()))
        else:
            return torch.device("cpu")

    def predict(self, image):
        if self.model is None:
            raise ModelNotLoadedError("Model has not been loaded. Call load() first.")

        return self.model(image)

    @staticmethod
    def get_results(results):
        (bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_name, confidence) = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        results = results.pandas().xyxy[0].to_dict(orient="records")

        if results:
            for result in results:
                confidence = result["confidence"]
                class_name = result["class"]
                bbox_x1 = int(result["xmin"])
                bbox_y1 = int(result["ymin"])
                bbox_x2 = int(result["xmax"])
                bbox_y2 = int(result["ymax"])

        return bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_name, confidence
