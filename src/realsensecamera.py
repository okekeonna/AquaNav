"""Live crab detection on the Intel RealSense color stream.

Shows the camera feed with bounding boxes around detected crabs. The model
runs in a background thread on the newest frame, so the video stays at the
full camera frame rate while boxes update as fast as the model allows
(about 4-5 times per second on CPU).

Usage:
    python src/realsensecamera.py
    python src/realsensecamera.py --conf 0.4
Press q or Esc in the video window to quit.
"""

import argparse
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

from test_model import calibrate, resolve

WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW = "Looking for crabs..."


class CrabDetector(threading.Thread):
    """Runs the model on the most recent frame in a background thread."""

    def __init__(self, model, conf: float):
        super().__init__(daemon=True)
        self.model = model
        self.conf = conf
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.stopped = threading.Event()
        self.frame = None
        self.detections = sv.Detections.empty()
        self.detect_fps = 0.0

    def submit(self, bgr_image: np.ndarray) -> None:
        """Hand over the newest frame. Older frames not yet processed are dropped."""
        with self.lock:
            self.frame = bgr_image
        self.new_frame.set()

    def latest(self):
        with self.lock:
            return self.detections, self.detect_fps

    def run(self) -> None:
        while not self.stopped.is_set():
            if not self.new_frame.wait(timeout=0.1):
                continue
            with self.lock:
                frame, self.frame = self.frame, None
                self.new_frame.clear()
            if frame is None:
                continue
            start = time.perf_counter()
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self.model.predict(rgb, threshold=self.conf)
            elapsed = time.perf_counter() - start
            with self.lock:
                self.detections = detections
                self.detect_fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def stop(self) -> None:
        self.stopped.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    print("Loading crab model...")
    model = RFDETRSmall(pretrain_weights=str(resolve(args.weights)), num_classes=3)
    names = calibrate(model)
    print("Class map:", {k: names[k] for k in sorted(names)})

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, smart_position=True)

    detector = CrabDetector(model, args.conf)
    detector.start()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("Begin realsense pipeline: ")
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("no frame")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            detector.submit(color_image.copy())

            detections, detect_fps = detector.latest()
            labels = [f"{names.get(int(c), f'class_{int(c)}')} {s:.2f}"
                      for c, s in zip(detections.class_id, detections.confidence)]
            annotated = box_annotator.annotate(color_image.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

            status = f"Crabs: {len(detections)}  |  detect {detect_fps:.1f} fps"
            cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
    

            cv2.imshow(WINDOW, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button

    finally:
        detector.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("done")


if __name__ == "__main__":
    main()
