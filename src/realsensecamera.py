"""Live crab detection on the Intel RealSense color stream.

Shows the camera feed with bounding boxes around detected crabs. The model
runs in a background thread on the newest frame, so the video stays at the
full camera frame rate while boxes update as fast as the model allows
(about 4-5 times per second on CPU).

Each crab's distance from the camera is measured from the depth stream, which
is aligned to the color image. The distance is the median depth over a central
region of the bounding box (not a single centre pixel), ignoring pixels with no
depth reading, so holes and noisy pixels don't throw it off.

Usage:
    python src/realsensecamera.py
    python src/realsensecamera.py --conf 0.4
Press q or Esc in the video window to quit.
"""

import argparse
import threading
import time
from dataclasses import dataclass

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

# Fraction of the box width/height sampled for depth, centred on the box.
DEPTH_REGION = 0.3
# Minimum share of valid (non-zero) depth pixels needed to trust a reading.
MIN_VALID_FRACTION = 0.2


def box_distance(depth_image: np.ndarray, box, depth_scale: float):
    """Median distance in metres over the central region of a box.

    Returns (distance or None, (x1, y1, x2, y2) of the sampled region).
    """
    h, w = depth_image.shape
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) * DEPTH_REGION / 2, 2)
    half_h = max((y2 - y1) * DEPTH_REGION / 2, 2)
    rx1, rx2 = int(max(cx - half_w, 0)), int(min(cx + half_w, w))
    ry1, ry2 = int(max(cy - half_h, 0)), int(min(cy + half_h, h))
    region = (rx1, ry1, rx2, ry2)

    patch = depth_image[ry1:ry2, rx1:rx2]
    valid = patch[patch > 0]  # 0 means no depth reading
    if patch.size == 0 or valid.size < MIN_VALID_FRACTION * patch.size:
        return None, region
    return float(np.median(valid)) * depth_scale, region


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


@dataclass
class CrabReading:
    """One detected crab with its measured distance."""
    name: str
    confidence: float
    box: tuple[float, float, float, float]
    distance: float | None  # metres along the depth axis; None if no valid depth
    # (x right, y down, z forward) in metres in the camera frame; None if no depth
    position: tuple[float, float, float] | None


class CrabCamera:
    """RealSense color + aligned depth stream with live crab detection."""

    def __init__(self, weights: str = "models/weights.pt", conf: float = 0.5):
        print("Loading crab model...")
        model = RFDETRSmall(pretrain_weights=str(resolve(weights)), num_classes=3)
        self.names = calibrate(model)
        print("Class map:", {k: self.names[k] for k in sorted(self.names)})
        self.detector = CrabDetector(model, conf)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, smart_position=True)
        self.pipeline = rs.pipeline()

    def start(self) -> None:
        self.detector.start()
        config = rs.config()
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

        print("Begin realsense pipeline: ")
        profile = self.pipeline.start(config)
        # Raw depth values are in device units; depth_scale converts them to metres.
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # Map depth pixels onto the color image so box coordinates line up.
        self.align = rs.align(rs.stream.color)
        self.intrinsics = (profile.get_stream(rs.stream.color)
                           .as_video_stream_profile().get_intrinsics())

    def read(self):
        """Wait for the next frame.

        Returns (annotated BGR image, list of CrabReading), or (None, []) if the
        frame was incomplete.
        """
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("no frame")
            return None, []

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        self.detector.submit(color_image.copy())

        detections, detect_fps = self.detector.latest()
        annotated = self.box_annotator.annotate(color_image.copy(), detections)
        crabs, labels = [], []
        for box, c, s in zip(detections.xyxy, detections.class_id, detections.confidence):
            distance, (rx1, ry1, rx2, ry2) = box_distance(depth_image, box, self.depth_scale)
            position = None
            if distance is not None:
                centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                position = tuple(rs.rs2_deproject_pixel_to_point(
                    self.intrinsics, centre, distance))
            name = self.names.get(int(c), f"class_{int(c)}")
            crabs.append(CrabReading(name, float(s), tuple(map(float, box)),
                                     distance, position))
            dist_text = f"{distance * 100:.1f} cm" if distance is not None else "-- cm"
            labels.append(f"{name} {s:.2f} | {dist_text}")
            # Show the region used for the depth reading.
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
        annotated = self.label_annotator.annotate(annotated, detections, labels)

        status = f"Crabs: {len(detections)}  |  detect {detect_fps:.1f} fps"
        cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        return annotated, crabs

    def stop(self) -> None:
        self.detector.stop()
        self.pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    camera = CrabCamera(args.weights, args.conf)
    camera.start()
    try:
        while True:
            annotated, _ = camera.read()
            if annotated is None:
                continue
            cv2.imshow(WINDOW, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("done")


if __name__ == "__main__":
    main()
