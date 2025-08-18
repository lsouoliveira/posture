import cv2


class Error(Exception):
    pass


class CouldNotOpenDeviceFound(Error):
    def __init__(self):
        super().__init__("Could not open the video device")


class FrameReadError(Error):
    def __init__(self):
        super().__init__("Could not read frame from video device")


class AlreadyOpenError(Error):
    def __init__(self):
        super().__init__("Camera is already open, cannot open again")


class Camera:
    def __init__(self):
        self.video_capture = None

    def open(self):
        if self.video_capture:
            raise AlreadyOpenError()

        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.video_capture.isOpened():
            raise CouldNotOpenDeviceFound()

    def capture(self):
        if self.video_capture is None:
            raise Exception("Video capture not started")

        ret, frame = self.video_capture.read()

        if not ret:
            raise FrameReadError()

        return frame

    def close(self):
        if not self.video_capture:
            return

        self.video_capture.release()
        self.video_capture = None
