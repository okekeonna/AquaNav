"""PyBullet prototype: show a crab seen by the RealSense in a 3D scene with the ROV.

The RealSense finds crabs (RF-DETR) and measures their 3D position from the
aligned depth stream. The nearest crab is placed in a 3D PyBullet world
relative to the simulated robot.

Modes:
  track     (default) The robot stays still and the crab marker moves to
            wherever the camera sees the crab, leaving a short fading trail.
            As the real crab moves, the simulated crab moves the same way.
  approach  Sense first, then move. The robot averages readings for a moment,
            turns to face the crab, then drives forward until it is
            --standoff metres away.

Usage:
    python src/crab_sim.py                      # track mode, real camera
    python src/crab_sim.py --mode approach --standoff 0.2
    python src/crab_sim.py --fake               # no camera: scripted crab readings

Distances are shown in centimetres (to 0.1 cm). In track mode the view zooms in
on the crab, which sits on a 1 cm grid (brighter lines every 5 cm) fixed where
the crab was first seen, so small movements are easy to see.

Keys (in either window): r = re-sense / reset, z = close-up / overview,
c = toggle auto camera view, q or Esc = quit.
"""

import argparse
import math
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from realsensecamera import CrabReading

# Robot geometry (metres). The camera sits on the front face of the hull.
ROBOT_SIZE = (0.40, 0.30, 0.20)
CAMERA_OFFSET = ROBOT_SIZE[0] / 2
START_POSITION = np.array([0.0, 0.0, 0.5])
CRAB_SIZE = (0.12, 0.10, 0.04)

MAX_SPEED = 0.25          # m/s
TURN_RATE = math.radians(60)  # rad/s
SMOOTHING = 0.3           # weight of each new reading in track mode
TRAIL_STEP = 0.002        # m the crab must move before a trail segment is drawn
TRAIL_SECONDS = 5.0       # how long trail segments stay visible
GRID_HALF_SIZE = 0.15     # m; the reference grid spans 30 x 30 cm around the crab
GRID_STEP = 0.01          # 1 cm grid lines (every 5th line is brighter)
CLOSE_UP_DISTANCE = 0.6   # m from the view camera to the crab in close-up view
CM = 100                  # metres -> centimetres, for display only
LOST_AFTER = 1.0          # s without a valid reading before the crab is "lost"
SENSE_TIME = 1.0          # s of readings averaged in approach mode
MIN_SENSE_SAMPLES = 10
WINDOW = "Crab camera"


def camera_to_body(position) -> np.ndarray:
    """RealSense frame (x right, y down, z forward) -> robot body frame
    (x forward, y left, z up), measured from the robot centre."""
    x, y, z = position
    return np.array([z + CAMERA_OFFSET, -x, -y])


def rotate(vector: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = vector
    return np.array([c * x - s * y, s * x + c * y, z])


def step_towards(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - current
    dist = float(np.linalg.norm(delta))
    return target.copy() if dist <= max_step else current + delta / dist * max_step


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def nearest_crab(crabs: list[CrabReading]) -> CrabReading | None:
    valid = [c for c in crabs if c.position is not None]
    return min(valid, key=lambda c: c.distance) if valid else None


# --------------------------------------------------------------------------- sources

class FakeCrabSource:
    """Scripted readings for testing without a camera: a crab drifting between
    1.3 m and 0.5 m away, with noise and occasional missing depth."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def read(self):
        time.sleep(1 / 30)
        t = time.perf_counter() - self.t0
        distance = 0.9 + 0.4 * math.cos(2 * math.pi * t / 20) + self.rng.normal(0, 0.01)
        x = 0.10 * math.sin(2 * math.pi * t / 14)
        y = 0.05
        image = np.zeros((120, 360, 3), np.uint8)
        if self.rng.random() < 0.1:  # simulated depth dropout
            cv2.putText(image, "FAKE CAMERA  depth: --", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return image, [CrabReading("Fake crab", 0.9, (0, 0, 0, 0), None, None)]
        cv2.putText(image, f"FAKE CAMERA  {distance:.2f} m", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image, [CrabReading("Fake crab", 0.9, (0, 0, 0, 0), distance, (x, y, distance))]

    def stop(self) -> None:
        pass


# --------------------------------------------------------------------------- world

class CrabWorld:
    """Kinematic PyBullet scene: seafloor, ROV and crab marker."""

    def __init__(self, gui: bool = True):
        if gui:
            p.connect(p.GUI, options="--background_color_red=0.03 "
                                     "--background_color_green=0.18 "
                                     "--background_color_blue=0.32")
        else:
            p.connect(p.DIRECT)  # headless, for tests
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        floor = p.loadURDF("plane.urdf")
        p.changeVisualShape(floor, -1, rgbaColor=[0.76, 0.70, 0.50, 1])

        half = [s / 2 for s in ROBOT_SIZE]
        self.robot = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=[1.0, 0.8, 0.1, 1]),
            linkMasses=[0],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.06, 0.03], rgbaColor=[0.1, 0.1, 0.1, 1])],
            linkPositions=[[half[0] + 0.02, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]],
        )
        self.crab = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[s / 2 for s in CRAB_SIZE],
                rgbaColor=[0.85, 0.2, 0.1, 1]),
            basePosition=[0, 0, -10])  # hidden until first seen
        self.line_id = -1
        self.text_id = -1
        self.follow_camera = True
        self.close_up = True
        self.grid_ids = []
        self.trail_point = None

    def reset(self) -> None:
        for item in self.grid_ids:
            p.removeUserDebugItem(item)
        self.grid_ids = []
        self.trail_point = None

    def draw_grid(self, centre: np.ndarray) -> None:
        """1 cm reference grid level with the underside of the crab."""
        z = centre[2] - CRAB_SIZE[2] / 2
        steps = round(GRID_HALF_SIZE / GRID_STEP)
        for i in range(-steps, steps + 1):
            major = i % 5 == 0
            colour = [0.9, 0.9, 0.9] if major else [0.45, 0.55, 0.6]
            width = 2 if major else 1
            off = i * GRID_STEP
            for a, b in (([off, -GRID_HALF_SIZE], [off, GRID_HALF_SIZE]),
                         ([-GRID_HALF_SIZE, off], [GRID_HALF_SIZE, off])):
                self.grid_ids.append(p.addUserDebugLine(
                    [centre[0] + a[0], centre[1] + a[1], z],
                    [centre[0] + b[0], centre[1] + b[1], z], colour, width))

    def update(self, robot_pos, robot_yaw, crab_pos, status: str) -> None:
        p.resetBasePositionAndOrientation(
            self.robot, robot_pos.tolist(), p.getQuaternionFromEuler([0, 0, robot_yaw]))
        if crab_pos is not None:
            if not self.grid_ids:
                self.draw_grid(crab_pos)
            p.resetBasePositionAndOrientation(self.crab, crab_pos.tolist(), [0, 0, 0, 1])
            # Fading trail of where the crab has been.
            if self.trail_point is None:
                self.trail_point = crab_pos.copy()
            elif np.linalg.norm(crab_pos - self.trail_point) > TRAIL_STEP:
                p.addUserDebugLine(self.trail_point.tolist(), crab_pos.tolist(),
                                   [1.0, 0.5, 0.2], 2, lifeTime=TRAIL_SECONDS)
                self.trail_point = crab_pos.copy()
            camera_pos = robot_pos + rotate(np.array([CAMERA_OFFSET, 0, 0]), robot_yaw)
            self.line_id = p.addUserDebugLine(
                camera_pos.tolist(), crab_pos.tolist(), [0.3, 1.0, 0.3], 2,
                replaceItemUniqueId=self.line_id)
        close_up = self.close_up and crab_pos is not None
        text_pos = crab_pos + [0, 0, 0.08] if close_up else robot_pos + [0, 0, 0.35]
        self.text_id = p.addUserDebugText(
            status, text_pos.tolist(), [1, 1, 1], 1.0 if close_up else 1.3,
            replaceItemUniqueId=self.text_id)

        if self.follow_camera:
            if close_up:
                p.resetDebugVisualizerCamera(cameraDistance=CLOSE_UP_DISTANCE, cameraYaw=-50,
                                             cameraPitch=-45,
                                             cameraTargetPosition=crab_pos.tolist())
            else:
                target = robot_pos if crab_pos is None else (robot_pos + crab_pos) / 2
                p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=-50, cameraPitch=-30,
                                             cameraTargetPosition=target.tolist())

    def pressed_keys(self) -> set[int]:
        return {k for k, state in p.getKeyboardEvents().items()
                if state & p.KEY_WAS_TRIGGERED}


# --------------------------------------------------------------------------- controllers

class TrackController:
    """Robot stays still; the crab is placed wherever the camera sees it."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.robot_pos = START_POSITION.copy()
        self.robot_yaw = 0.0
        self.crab_world = None   # current crab position in the world
        self.first_offset = None  # body-frame offset when the crab was first seen
        self.smoothed = None     # smoothed body-frame offset to the crab
        self.last_seen = None

    def update(self, reading: CrabReading | None, dt: float, now: float) -> str:
        if reading is not None:
            offset = camera_to_body(reading.position)
            self.smoothed = offset if self.smoothed is None else (
                SMOOTHING * offset + (1 - SMOOTHING) * self.smoothed)
            if self.first_offset is None:
                self.first_offset = offset
            self.last_seen = now
            self.crab_world = self.robot_pos + rotate(self.smoothed, self.robot_yaw)

        if self.crab_world is None:
            return "Searching for crab..."
        if now - self.last_seen > LOST_AFTER:
            return "LOST - showing last position"
        # Depth = straight ahead from the camera; range = straight-line distance.
        depth = self.smoothed[0] - CAMERA_OFFSET
        crab_range = float(np.linalg.norm(self.smoothed - [CAMERA_OFFSET, 0, 0]))
        # Movement since first seen, along the robot's axes.
        fwd, left, up = (self.smoothed - self.first_offset) * CM
        return (f"depth {depth * CM:.1f} cm | range {crab_range * CM:.1f} cm | "
                f"moved fwd {fwd:+.1f} left {left:+.1f} up {up:+.1f} cm")


class ApproachController:
    """Sense the crab, then turn and drive to a standoff distance."""

    def __init__(self, standoff: float):
        self.standoff = standoff
        self.reset()

    def reset(self) -> None:
        self.robot_pos = START_POSITION.copy()
        self.robot_yaw = 0.0
        self.crab_world = None
        self.samples = []
        self.sense_start = None
        self.state = "SENSING"

    def camera_pos(self) -> np.ndarray:
        return self.robot_pos + rotate(np.array([CAMERA_OFFSET, 0, 0]), self.robot_yaw)

    def update(self, reading: CrabReading | None, dt: float, now: float) -> str:
        if self.state == "SENSING":
            if reading is not None:
                self.sense_start = self.sense_start or now
                self.samples.append(camera_to_body(reading.position))
            if (self.sense_start and now - self.sense_start >= SENSE_TIME
                    and len(self.samples) >= MIN_SENSE_SAMPLES):
                offset = np.median(np.array(self.samples), axis=0)
                self.crab_world = self.robot_pos + rotate(offset, self.robot_yaw)
                self.state = "TURNING"
            return f"SENSING ({len(self.samples)} readings)"

        to_crab = self.crab_world - self.robot_pos
        if self.state == "TURNING":
            heading = math.atan2(to_crab[1], to_crab[0])
            error = angle_diff(heading, self.robot_yaw)
            turn = max(-TURN_RATE * dt, min(TURN_RATE * dt, error))
            self.robot_yaw += turn
            if abs(error) < math.radians(1):
                self.robot_yaw = heading
                direction = to_crab / np.linalg.norm(to_crab)
                self.goal = self.crab_world - direction * (self.standoff + CAMERA_OFFSET)
                self.state = "DRIVING"
            return f"TURNING {math.degrees(error):+.0f} deg"

        if self.state == "DRIVING":
            self.robot_pos = step_towards(self.robot_pos, self.goal, MAX_SPEED * dt)
            if np.linalg.norm(self.goal - self.robot_pos) < 1e-3:
                self.state = "ARRIVED"

        gap = float(np.linalg.norm(self.crab_world - self.camera_pos()))
        moved = float(np.linalg.norm(self.robot_pos - START_POSITION))
        suffix = "  (press r to sense again)" if self.state == "ARRIVED" else ""
        return f"{self.state} | gap {gap * CM:.1f} cm | moved {moved * CM:.1f} cm{suffix}"


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["track", "approach"], default="track")
    parser.add_argument("--standoff", type=float, default=0.15,
                        help="approach mode: stop this far from the crab (m)")
    parser.add_argument("--fake", action="store_true", help="use scripted readings, no camera")
    parser.add_argument("--weights", default="models/weights.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    if args.fake:
        source = FakeCrabSource()
    else:
        from realsensecamera import CrabCamera
        source = CrabCamera(args.weights, args.conf)

    controller = (ApproachController(args.standoff) if args.mode == "approach"
                  else TrackController())
    world = CrabWorld()
    source.start()
    last = time.perf_counter()
    try:
        while p.isConnected():
            image, crabs = source.read()
            now = time.perf_counter()
            dt, last = now - last, now

            status = controller.update(nearest_crab(crabs), dt, now)
            world.update(controller.robot_pos, controller.robot_yaw,
                         controller.crab_world, status)

            keys = world.pressed_keys()
            if image is not None:
                # Status also drawn on the camera feed, where it is always readable.
                bar = np.zeros((30, image.shape[1], 3), np.uint8)
                cv2.putText(bar, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, np.vstack([image, bar]))
            cv_key = cv2.waitKey(1) & 0xFF
            if cv_key != 255:
                keys.add(cv_key)
            if keys & {ord("q"), 27}:
                break
            if ord("r") in keys:
                controller.reset()
                world.reset()
            if ord("z") in keys:
                world.close_up = not world.close_up
            if ord("c") in keys:
                world.follow_camera = not world.follow_camera
    except p.error:
        pass  # simulation window was closed
    finally:
        source.stop()
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()
        print("done")


if __name__ == "__main__":
    main()
