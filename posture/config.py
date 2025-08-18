class Config:
    def __init__(self):
        self.MODEL_PATH = "./data/models/small640.pt"
        self.MODEL_CONFIDENCE_THRESHOLD = 0.5
        self.MODEL_IOU_THRESHOLD = 0.5
        self.MODEL_CLASSES = [0, 1]
        self.MODEL_AGNOSTIC = False
        self.MODEL_MULTI_LABEL = False
        self.MODEL_MAX_DET = 1
        self.MODEL_AMP = True

        self.MONITOR_INTERVAL = 2

        self.LOG_LEVEL = "CRITICAL"


settings = Config()
