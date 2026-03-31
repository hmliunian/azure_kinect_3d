"""Main application: RGBD camera + SAM3 segmentation + 3D reconstruction.

Interactive OpenCV GUI workflow:
  1. Live camera preview
  2. Press SPACE to freeze frame
  3. Draw box (mouse drag) or type text to segment
  4. Press 's' to save point cloud + mesh
  5. Press 'r' to reset and continue
  6. Press 'q' to quit
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from azure_kinect_3d.camera import RealSenseCamera, CaptureResult
from azure_kinect_3d.segmentor import Sam3Segmentor
from azure_kinect_3d.reconstruction import (
    Sam3DReconstructor,
    save_reconstruction,
)
from azure_kinect_3d.deploy import deploy_to_discoverse

# ── GUI State ──────────────────────────────────────────────────────────────────

class AppState:
    # Modes
    LIVE = "live"           # Showing live camera feed
    FROZEN = "frozen"       # Frame frozen, ready for segmentation
    SEGMENTED = "segmented" # Segmentation result shown

    def __init__(self):
        self.mode = self.LIVE
        self.frozen_frame: CaptureResult | None = None
        self.mask: np.ndarray | None = None  # (H, W) bool
        self.display_image: np.ndarray | None = None

        # Box drawing state
        self.drawing = False
        self.box_start = None
        self.box_end = None

        # Point click state
        self.click_points: list[tuple[int, int, bool]] = []  # (x, y, positive)

        # Text prompt
        self.text_prompt = ""
        self.entering_text = False

        self.save_count = 0
        self.last_saved_mesh: str | None = None


# ── Drawing Helpers ────────────────────────────────────────────────────────────

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Overlay a colored semi-transparent mask on an image."""
    out = image.copy()
    overlay = np.zeros_like(image)
    overlay[mask] = color
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    # Draw mask contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_status_bar(image: np.ndarray, text: str, color=(50, 50, 50)) -> np.ndarray:
    """Draw a status bar at the bottom of the image."""
    h, w = image.shape[:2]
    bar_h = 40
    out = image.copy()
    cv2.rectangle(out, (0, h - bar_h), (w, h), color, -1)
    cv2.putText(out, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def load_from_input(input_dir: str) -> tuple[CaptureResult, np.ndarray | None] | None:
    """Load the latest saved capture from the input directory.

    Looks for files matching the pattern: {prefix}_rgb.png, {prefix}_pointmap.npy,
    {prefix}_mask.npy (optional).

    Returns:
        (CaptureResult, mask) tuple, or None if no valid data found.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"[App] Input directory not found: {input_dir}")
        return None

    # Find all rgb files and pick the latest by name (timestamp-sorted)
    rgb_files = sorted(input_path.glob("*_rgb.png"), reverse=True)
    if not rgb_files:
        print(f"[App] No *_rgb.png files found in {input_dir}")
        return None

    rgb_path = rgb_files[0]
    prefix = rgb_path.name.rsplit("_rgb.png", 1)[0]
    print(f"[App] Loading from input: {prefix}")

    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        print(f"[App] Failed to read {rgb_path}")
        return None

    pointmap = None
    pointmap_path = input_path / f"{prefix}_pointmap.npy"
    if pointmap_path.exists():
        pointmap = np.load(str(pointmap_path))
        print(f"[App] Loaded pointmap: {pointmap.shape}")

    mask = None
    mask_path = input_path / f"{prefix}_mask.npy"
    if mask_path.exists():
        mask = np.load(str(mask_path)).astype(bool)
        print(f"[App] Loaded mask: {mask.shape}, {mask.sum()} px")

    capture = CaptureResult(rgb=rgb, pointmap=pointmap)
    return capture, mask


def draw_help_overlay(image: np.ndarray, has_camera: bool = True) -> np.ndarray:
    """Show key bindings on the image."""
    out = image.copy()
    lines = [
        "L-click: Point select (fg)",
        "R-click: Point select (bg)",
        "Mouse drag: Box select",
        "t: Text prompt",
        "s: Save 3D output",
        "p: Deploy to sim",
        "r: Reset segmentation",
        "q: Quit",
    ]
    if has_camera:
        lines.insert(0, "SPACE: Freeze/Unfreeze")
    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 22
    return out


# ── Mouse Callback ─────────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    state: AppState = param

    if state.mode != AppState.FROZEN and state.mode != AppState.SEGMENTED:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.box_start = (x, y)
        state.box_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.box_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        state.box_end = (x, y)

        if state.box_start is not None:
            x0 = min(state.box_start[0], state.box_end[0])
            y0 = min(state.box_start[1], state.box_end[1])
            x1 = max(state.box_start[0], state.box_end[0])
            y1 = max(state.box_start[1], state.box_end[1])

            if (x1 - x0) > 10 and (y1 - y0) > 10:
                # Large drag → box prompt
                state._pending_box = [x0, y0, x1, y1]
            else:
                # Small drag / click → positive point prompt
                cx = (state.box_start[0] + state.box_end[0]) // 2
                cy = (state.box_start[1] + state.box_end[1]) // 2
                state.click_points.append((cx, cy, True))
                state._pending_point = (cx, cy, True)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click → negative (background) point
        state.click_points.append((x, y, False))
        state._pending_point = (x, y, False)


# ── Main Application ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RealSense D405 + SAM3 Segmentation + 3D Reconstruction")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number")
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM3 checkpoint path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--input-dir", type=str, default="input", help="Fallback input directory when no camera")
    parser.add_argument("--no-sam", action="store_true", help="Skip SAM3 loading (camera test only)")
    args = parser.parse_args()

    # Resolve checkpoint path
    project_root = Path(__file__).resolve().parent.parent.parent
    if args.checkpoint is None:
        args.checkpoint = str(project_root / "third_party" / "sam3" / "ckp" / "sam3.pt")
    output_dir = str(project_root / args.output_dir)
    input_dir = str(project_root / args.input_dir)

    # ── Init Camera (fallback to input files) ──
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        serial_number=args.serial,
    )
    has_camera = camera.start()
    fallback_capture = None
    fallback_mask = None

    if not has_camera:
        print("[App] No camera detected, loading from input directory ...")
        loaded = load_from_input(input_dir)
        if loaded is None:
            print("[App] No input data available either. Exiting.")
            return
        fallback_capture, fallback_mask = loaded

    # ── Init Segmentor (eager load) ──
    segmentor = None
    if not args.no_sam:
        segmentor = Sam3Segmentor(checkpoint_path=args.checkpoint)
        segmentor.load_model()

    # ── Init 3D Reconstructor (lazy load) ──
    reconstructor = Sam3DReconstructor()

    # ── GUI ──
    state = AppState()

    # If no camera, pre-load the fallback data as a frozen frame
    if not has_camera:
        state.frozen_frame = fallback_capture
        if fallback_mask is not None:
            state.mask = fallback_mask
            state.mode = AppState.SEGMENTED
        else:
            state.mode = AppState.FROZEN

    win_name = "RGBD + SAM3 Segmentation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, state)

    print("\n=== Controls ===")
    if has_camera:
        print("SPACE : Freeze / unfreeze frame")
    else:
        print("(No camera - using input files)")
    print("L-click: Point select (foreground)")
    print("R-click: Point select (background)")
    print("Drag  : Draw box to segment")
    print("t     : Enter text prompt")
    print("s     : Save point cloud + mesh")
    print("p     : Deploy to DISCOVERSE sim")
    print("r     : Reset segmentation")
    print("h     : Toggle help overlay")
    print("q     : Quit")
    print("================\n")

    show_help = True

    while True:
        # ── Capture / Display ──
        if state.mode == AppState.LIVE:
            if not has_camera:
                time.sleep(0.01)
                continue
            result = camera.capture()
            if result is None:
                time.sleep(0.01)
                continue
            display = result.rgb.copy()
            status = "[LIVE] SPACE=freeze | q=quit | Depth: ON"

        elif state.mode in (AppState.FROZEN, AppState.SEGMENTED):
            display = state.frozen_frame.rgb.copy()

            # Show mask overlay
            if state.mask is not None:
                display = overlay_mask(display, state.mask)
                n_pts = state.mask.sum()
                status = f"[SEGMENTED] {n_pts} px | s=save | r=reset | Draw box for new"
            else:
                status = "[FROZEN] Draw box or press 't' for text prompt"

            # Show box being drawn
            if state.drawing and state.box_start and state.box_end:
                cv2.rectangle(display, state.box_start, state.box_end, (0, 255, 0), 2)

            # Show clicked points
            for px, py, positive in state.click_points:
                color = (0, 255, 0) if positive else (0, 0, 255)
                cv2.circle(display, (px, py), 6, color, -1)
                cv2.circle(display, (px, py), 6, (255, 255, 255), 1)

        # ── Text input mode ──
        if state.entering_text:
            status = f"[TEXT] Type prompt: {state.text_prompt}_ | ENTER=submit | ESC=cancel"

        display = draw_status_bar(display, status)
        if show_help:
            display = draw_help_overlay(display, has_camera=has_camera)

        cv2.imshow(win_name, display)

        # ── Process pending box segmentation ──
        if hasattr(state, "_pending_box") and state._pending_box is not None:
            box = state._pending_box
            state._pending_box = None
            if segmentor is not None:
                print(f"[App] Segmenting with box: {box}")
                if state.mode == AppState.FROZEN:
                    segmentor.set_image(state.frozen_frame.rgb)
                segmentor.reset()
                segmentor.set_image(state.frozen_frame.rgb)
                segmentor.add_box(box, positive=True)
                mask = segmentor.get_mask()
                if mask is not None:
                    state.mask = mask
                    state.mode = AppState.SEGMENTED
                    scores = segmentor.get_scores()
                    print(f"[App] Found {segmentor.num_detections} objects, scores: {scores}")
                else:
                    print("[App] No objects detected in box region")
            else:
                print("[App] SAM3 not loaded (use --no-sam to skip)")

        # ── Process pending point segmentation ──
        if hasattr(state, "_pending_point") and state._pending_point is not None:
            px, py, positive = state._pending_point
            state._pending_point = None
            if segmentor is not None:
                label = "fg" if positive else "bg"
                print(f"[App] Segmenting with point ({px}, {py}) [{label}]")
                if not segmentor._image_set:
                    segmentor.set_image(state.frozen_frame.rgb)
                segmentor.add_point([px, py], positive=positive)
                mask = segmentor.get_mask()
                if mask is not None:
                    state.mask = mask
                    state.mode = AppState.SEGMENTED
                    scores = segmentor.get_scores()
                    print(f"[App] Found {segmentor.num_detections} objects, scores: {scores}")
                else:
                    print("[App] No objects detected at point")
            else:
                print("[App] SAM3 not loaded (use --no-sam to skip)")

        # ── Key handling ──
        key = cv2.waitKey(30) & 0xFF

        if state.entering_text:
            if key == 27:  # ESC
                state.entering_text = False
                state.text_prompt = ""
            elif key == 13:  # ENTER
                if state.text_prompt and segmentor is not None:
                    print(f"[App] Segmenting with text: '{state.text_prompt}'")
                    segmentor.set_image(state.frozen_frame.rgb)
                    segmentor.segment_by_text(state.text_prompt)
                    mask = segmentor.get_mask()
                    if mask is not None:
                        state.mask = mask
                        state.mode = AppState.SEGMENTED
                        print(f"[App] Found {segmentor.num_detections} objects")
                    else:
                        print("[App] No objects detected for text prompt")
                state.entering_text = False
                state.text_prompt = ""
            elif key == 8:  # BACKSPACE
                state.text_prompt = state.text_prompt[:-1]
            elif 32 <= key <= 126:  # Printable ASCII
                state.text_prompt += chr(key)
            continue

        if key == ord("q"):
            break

        elif key == ord(" "):  # SPACE
            if not has_camera:
                continue  # No camera, SPACE does nothing
            if state.mode == AppState.LIVE:
                result = camera.capture()
                if result is not None:
                    state.frozen_frame = result
                    state.mode = AppState.FROZEN
                    state.mask = None
                    print("[App] Frame frozen.")
            else:
                state.mode = AppState.LIVE
                state.mask = None
                state.frozen_frame = None
                state.click_points.clear()
                if segmentor:
                    segmentor.reset()
                print("[App] Back to live.")

        elif key == ord("t"):
            if state.mode in (AppState.FROZEN, AppState.SEGMENTED):
                state.entering_text = True
                state.text_prompt = ""

        elif key == ord("s"):
            if state.mode in (AppState.FROZEN, AppState.SEGMENTED):
                frame = state.frozen_frame
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = f"seg_{timestamp}"

                # Save pointmap
                if frame.pointmap is not None:
                    np.save(str(out / f"{prefix}_pointmap.npy"), frame.pointmap)
                    print(f"[App] Saved pointmap: {prefix}_pointmap.npy "
                          f"{frame.pointmap.shape}")
                else:
                    print("[App] No pointmap available.")

                # Also save RGB and mask for reference
                cv2.imwrite(str(out / f"{prefix}_rgb.png"), frame.rgb)
                if state.mask is not None:
                    np.save(str(out / f"{prefix}_mask.npy"), state.mask)

                state.save_count += 1
                print(f"[App] Saved #{state.save_count}")

        elif key == ord("p"):
            if state.last_saved_mesh and os.path.exists(state.last_saved_mesh):
                import threading
                print(f"[App] Deploying {state.last_saved_mesh} to DISCOVERSE ...")
                threading.Thread(
                    target=deploy_to_discoverse,
                    args=(state.last_saved_mesh,),
                    daemon=True,
                ).start()
            else:
                print("[App] No saved mesh to deploy. Press 's' first.")

        elif key == ord("r"):
            state.mask = None
            state.click_points.clear()
            if state.frozen_frame is not None:
                state.mode = AppState.FROZEN
            if segmentor:
                segmentor.reset()
            print("[App] Reset segmentation.")

        elif key == ord("h"):
            show_help = not show_help

    if has_camera:
        camera.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
