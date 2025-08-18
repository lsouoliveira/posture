import sys
from PyQt5.QtWidgets import QApplication
import argparse

from posture.application import Application
from posture.visualizer.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Posture Visualizer Application")
    parser.add_argument(
        "--visualizer",
        action="store_true",
        help="Run the posture visualizer application",
    )
    return parser.parse_args()


def start_visualizer():
    app = QApplication(sys.argv)
    visualizer = Visualizer()
    visualizer.show()
    sys.exit(app.exec_())


def start_monitor():
    app = Application()
    app.init()
    app.run()


if __name__ == "__main__":
    args = parse_args()

    if args.visualizer:
        start_visualizer()
    else:
        start_monitor()
