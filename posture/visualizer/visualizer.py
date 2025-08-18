from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
import cv2

from posture.detector.detector import Detector
from posture.detector.inference_model import InferenceModel
from posture.camera import Camera
from posture.config import settings


def draw_rectangle(image, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text, position, color=(0, 255, 0), font_scale=0.5, thickness=1):
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )


def frame_to_qt_image(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w

    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


class Visualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.camera = Camera()
        self._init()

    def _init(self):
        self._setup_ui()
        self._setup_detector()
        self._setup_timer()
        self.camera.open()

    def _setup_ui(self):
        self.setWindowTitle("Posture - Visualizer")
        self.resize(800, 600)

        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.preview)

        self.setLayout(layout)

    def _setup_detector(self):
        self.model = InferenceModel(settings.MODEL_PATH).load()
        self.detector = Detector(self.model)

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

    def _update_frame(self):
        frame = self.camera.capture()
        posture = self.detector.detect(frame)

        if posture:
            x1, y1, x2, y2 = posture.bounding_box
            draw_rectangle(frame, (x1, y1, x2, y2))
            draw_text(
                frame,
                f"{posture.posture_type.value} ({posture.confidence:.2f})",
                (x1, y1 - 10),
            )

        pixmap = QPixmap.fromImage(frame_to_qt_image(frame)).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio
        )

        self.preview.setPixmap(pixmap)

    def closeEvent(self, event):
        self.camera.close()

        event.accept()
