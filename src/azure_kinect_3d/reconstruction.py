"""3D reconstruction using SAM3D Objects pipeline.

Takes RGBD image + segmentation mask, runs neural 3D reconstruction
to produce a complete mesh and gaussian splat point cloud.

SAM3D runs in its own Python venv (different torch/CUDA version) via subprocess.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ── SAM3D Objects Subprocess Bridge ──────────────────────────────────────────

SAM3D_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "sam3d_object"
SAM3D_PYTHON = SAM3D_ROOT / ".venv" / "bin" / "python"
BRIDGE_SCRIPT = SAM3D_ROOT / "bridge_reconstruct.py"


class Sam3DReconstructor:
    """Neural 3D reconstruction using SAM3D Objects pipeline.

    Runs SAM3D in a separate subprocess using sam3d_object's own venv
    to avoid dependency conflicts (cu121 vs cu130, Python 3.11 vs 3.12).
    """

    def __init__(self):
        if not SAM3D_PYTHON.exists():
            raise FileNotFoundError(
                f"SAM3D venv not found: {SAM3D_PYTHON}\n"
                f"Ensure third_party/sam3d_object/.venv symlink is valid."
            )

    def reconstruct(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        pointmap: np.ndarray,
        output_dir: str,
        prefix: str = "object",
        seed: int = 42,
    ) -> dict | None:
        """Run SAM3D reconstruction via subprocess.

        Args:
            rgb: (H, W, 3) uint8 BGR image (OpenCV format from camera).
            mask: (H, W) bool segmentation mask.
            pointmap: (H, W, 3) float32 XYZ coordinates in meters.
            output_dir: Directory to save output files.
            prefix: Filename prefix for outputs.
            seed: Random seed for reproducibility.

        Returns:
            dict with keys:
                "vertices": (V, 3) float32
                "faces": (F, 3) int32
                "gs_points": (N, 3) float32
                "gs_colors": (N, 3) uint8
                "mesh_path": str (path to .obj)
                "pointcloud_path": str (path to .ply)
            Or None on failure.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Write inputs to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rgb_path = str(tmp / "rgb.png")
            mask_path = str(tmp / "mask.npy")
            pm_path = str(tmp / "pointmap.npy")

            cv2.imwrite(rgb_path, rgb)
            np.save(mask_path, mask)
            np.save(pm_path, pointmap)

            # Call bridge script in sam3d_object's venv
            cmd = [
                str(SAM3D_PYTHON),
                str(BRIDGE_SCRIPT),
                "--rgb", rgb_path,
                "--mask", mask_path,
                "--pointmap", pm_path,
                "--output-dir", str(out),
                "--prefix", prefix,
                "--seed", str(seed),
            ]

            print(f"[SAM3D] Running reconstruction subprocess...")
            result = subprocess.run(
                cmd,
                cwd=str(SAM3D_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )

            # Stream output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        print(f"  [stderr] {line}")

            if result.returncode != 0 or "SUCCESS" not in result.stdout:
                print(f"[SAM3D] Reconstruction failed (exit code {result.returncode})")
                return None

        # Load results
        mesh_path = str(out / f"{prefix}_mesh.obj")
        ply_path = str(out / f"{prefix}_pointcloud.ply")

        vertices = np.load(str(out / f"{prefix}_mesh_vertices.npy"))
        faces = np.load(str(out / f"{prefix}_mesh_faces.npy"))
        gs_points = np.load(str(out / f"{prefix}_gs_points.npy"))
        gs_colors = np.load(str(out / f"{prefix}_gs_colors.npy"))

        print(f"[SAM3D] Done: {mesh_path} ({len(vertices)} verts, {len(faces)} faces)")
        print(f"[SAM3D] Done: {ply_path} ({len(gs_points)} points)")

        return {
            "vertices": vertices,
            "faces": faces,
            "gs_points": gs_points,
            "gs_colors": gs_colors,
            "mesh_path": mesh_path,
            "pointcloud_path": ply_path,
        }


# ── Save Utilities ────────────────────────────────────────────────────────────


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None):
    """Save point cloud to PLY file.

    Args:
        path: Output file path.
        points: (N, 3) float32 XYZ.
        colors: (N, 3) uint8 RGB (optional).
    """
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for x, y, z in points:
                f.write(f"{x} {y} {z}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print(f"[Reconstruction] Saved point cloud: {path} ({len(points)} points)")


def save_mesh_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    """Save mesh to OBJ file.

    Args:
        path: Output file path.
        vertices: (V, 3) float32 vertices.
        faces: (F, 3) int32 face indices (0-indexed).
    """
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"[Reconstruction] Saved mesh: {path} "
          f"({len(vertices)} vertices, {len(faces)} faces)")


# ── Convenience Entry Point ──────────────────────────────────────────────────


def save_reconstruction(
    reconstructor: Sam3DReconstructor,
    output_dir: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    pointmap: np.ndarray,
    prefix: str = "object",
    seed: int = 42,
) -> str | None:
    """Run SAM3D reconstruction and save outputs.

    Args:
        reconstructor: Sam3DReconstructor instance.
        output_dir: Directory to save files.
        rgb: (H, W, 3) uint8 BGR image.
        mask: (H, W) bool segmentation mask.
        pointmap: (H, W, 3) float32 XYZ coordinates.
        prefix: Filename prefix.
        seed: Random seed.

    Returns:
        Path to saved .obj mesh file, or None on failure.
    """
    result = reconstructor.reconstruct(
        rgb, mask, pointmap,
        output_dir=output_dir,
        prefix=prefix,
        seed=seed,
    )
    if result is None:
        return None
    return result["mesh_path"]
