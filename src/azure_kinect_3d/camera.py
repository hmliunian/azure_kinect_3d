"""RGBD camera capture module using Intel RealSense.

Provides aligned RGB + Depth + Pointmap from RealSense cameras (D405, D435, D455).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrealsense2 as rs


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def depth_to_pointmap(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to 3D point map (H, W, 3) in meters."""
        h, w = depth.shape[:2]
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        z = depth.astype(np.float32)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.stack([x, y, z], axis=-1)


@dataclass
class CaptureResult:
    """A single capture from the camera."""

    rgb: np.ndarray  # (H, W, 3) uint8 BGR
    depth: Optional[np.ndarray] = None  # (H, W) float32, meters
    pointmap: Optional[np.ndarray] = None  # (H, W, 3) float32, XYZ in meters


class RealSenseCamera:
    """RealSense RGBD camera (D405, D435, D455).

    Captures aligned RGB + Depth streams and generates pointmaps.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial_number: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = 0.0001
        self.intrinsics: Optional[CameraIntrinsics] = None

    def start(self) -> bool:
        """Open the RealSense camera. Returns True on success."""
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()

            if self.serial_number:
                config.enable_device(self.serial_number)

            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            profile = self._pipeline.start(config)

            # Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

            # Align depth to color
            self._align = rs.align(rs.stream.color)

            # Extract intrinsics
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = depth_profile.get_intrinsics()
            self.intrinsics = CameraIntrinsics(
                fx=intr.fx, fy=intr.fy,
                cx=intr.ppx, cy=intr.ppy,
                width=intr.width, height=intr.height,
            )

            dev = profile.get_device()
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            print(f"[Camera] {name} (SN:{sn}) opened @ {self.width}x{self.height}")
            print(f"[Camera] Depth scale: {self._depth_scale}")
            print(f"[Camera] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

            # Let auto-exposure settle
            for _ in range(15):
                self._pipeline.wait_for_frames()

            return True

        except Exception as e:
            print(f"[Camera] Failed to start: {e}")
            self._pipeline = None
            return False

    def stop(self):
        """Stop the camera pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            print("[Camera] Stopped.")

    @property
    def is_running(self) -> bool:
        return self._pipeline is not None

    def capture(self) -> Optional[CaptureResult]:
        """Capture one aligned RGB+Depth frame with pointmap.

        Returns None on failure.
        """
        if self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            rgb = np.asanyarray(color_frame.get_data())  # BGR uint8
            depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
            depth = depth_raw.astype(np.float32) * self._depth_scale  # meters

            pointmap = self.intrinsics.depth_to_pointmap(depth)

            return CaptureResult(rgb=rgb, depth=depth, pointmap=pointmap)

        except Exception as e:
            print(f"[Camera] Capture error: {e}")
            return None

    def capture_rgb(self) -> Optional[np.ndarray]:
        """Convenience: capture BGR frame only."""
        result = self.capture()
        return result.rgb if result is not None else None
