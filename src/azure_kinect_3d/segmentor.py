"""SAM3 interactive segmentation module.

Wraps the SAM3 model for click/box/text based object segmentation.
"""

from typing import Optional

import numpy as np
import torch
import PIL.Image


class Sam3Segmentor:
    """Interactive segmentor using SAM3 (Segment Anything Model 3).

    Usage:
        seg = Sam3Segmentor(checkpoint_path="third_party/sam3/ckp/sam3.pt")
        seg.set_image(rgb_bgr)        # Set image (BGR uint8)
        seg.segment_by_text("cup")    # Text prompt
        seg.add_box([x0, y0, x1, y1]) # Or box prompt (pixel coords)
        mask = seg.get_mask()          # (H, W) bool
        seg.reset()
    """

    def __init__(
        self,
        checkpoint_path: str = "third_party/sam3/ckp/sam3.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.3,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        self._model = None
        self._processor = None
        self._state = None
        self._image_set = False

    def load_model(self):
        """Load SAM3 model (lazy, called on first use)."""
        if self._model is not None:
            return

        print("[SAM3] Loading model...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            load_from_HF=False,
        )

        self._processor = Sam3Processor(
            self._model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        print("[SAM3] Model loaded.")

    def set_image(self, image_bgr: np.ndarray):
        """Set the image for segmentation.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image (OpenCV format).
        """
        self.load_model()

        # Convert BGR to RGB PIL image
        image_rgb = image_bgr[:, :, ::-1].copy()
        pil_image = PIL.Image.fromarray(image_rgb)

        with torch.inference_mode():
            self._state = self._processor.set_image(pil_image)
        self._image_set = True

    def segment_by_text(self, prompt: str):
        """Run segmentation with a text prompt.

        Args:
            prompt: Text describing the object to segment (e.g. "cup", "person").
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        with torch.inference_mode():
            self._state = self._processor.set_text_prompt(prompt, self._state)

    def add_point(self, point_xy: list[float], positive: bool = True):
        """Add a point prompt for segmentation.

        Args:
            point_xy: [x, y] in pixel coordinates.
            positive: True for foreground, False for background.
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        x, y = point_xy
        img_h = self._state["original_height"]
        img_w = self._state["original_width"]

        # Normalize to [0, 1]
        nx = x / img_w
        ny = y / img_h

        if "language_features" not in self._state["backbone_out"]:
            dummy_text_outputs = self._model.backbone.forward_text(
                ["visual"], device=self.device
            )
            self._state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in self._state:
            self._state["geometric_prompt"] = self._model._get_dummy_prompt()

        points = torch.tensor([nx, ny], device=self.device, dtype=torch.float32).view(1, 1, 2)
        labels = torch.tensor([positive], device=self.device, dtype=torch.bool).view(1, 1)
        self._state["geometric_prompt"].append_points(points, labels)

        with torch.inference_mode():
            self._state = self._processor._forward_grounding(self._state)

    def add_box(self, box_xyxy: list[float], positive: bool = True):
        """Add a box prompt for segmentation.

        Args:
            box_xyxy: [x0, y0, x1, y1] in pixel coordinates.
            positive: True for include, False for exclude.
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        x0, y0, x1, y1 = box_xyxy
        img_h = self._state["original_height"]
        img_w = self._state["original_width"]

        # Convert xyxy pixels to cxcywh normalized [0,1]
        cx = (x0 + x1) / 2.0 / img_w
        cy = (y0 + y1) / 2.0 / img_h
        w = abs(x1 - x0) / img_w
        h = abs(y1 - y0) / img_h

        with torch.inference_mode():
            self._state = self._processor.add_geometric_prompt(
                [cx, cy, w, h], positive, self._state
            )

    def get_masks(self) -> list[np.ndarray]:
        """Get all detected masks as a list of (H, W) bool arrays."""
        if self._state is None or "masks" not in self._state:
            return []

        masks = self._state["masks"]
        return [m[0].cpu().numpy().astype(bool) for m in masks]

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the best (highest score) mask as (H, W) bool array."""
        masks = self.get_masks()
        if not masks:
            return None

        scores = self._state.get("scores", None)
        if scores is not None and len(scores) > 0:
            best_idx = scores.argmax().item()
            return masks[best_idx]
        return masks[0]

    def get_scores(self) -> list[float]:
        """Get confidence scores for each detected mask."""
        if self._state is None or "scores" not in self._state:
            return []
        return self._state["scores"].cpu().tolist()

    def get_boxes(self) -> list[np.ndarray]:
        """Get bounding boxes [x0, y0, x1, y1] for each detected mask."""
        if self._state is None or "boxes" not in self._state:
            return []
        return [b.cpu().numpy() for b in self._state["boxes"]]

    def reset(self):
        """Clear all prompts, keep the image."""
        if self._processor is not None and self._state is not None:
            self._processor.reset_all_prompts(self._state)
            # Remove prompted boxes tracking
            if "prompted_boxes" in self._state:
                del self._state["prompted_boxes"]

    @property
    def num_detections(self) -> int:
        if self._state is None or "masks" not in self._state:
            return 0
        return len(self._state["masks"])
