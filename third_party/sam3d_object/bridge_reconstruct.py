#!/usr/bin/env python3
"""Bridge script: runs SAM3D reconstruction inside sam3d_object's own venv.

Called via subprocess from the main project. Communicates through .npy files.

Usage:
    python bridge_reconstruct.py \
        --rgb /path/to/rgb.png \
        --mask /path/to/mask.npy \
        --pointmap /path/to/pointmap.npy \
        --output-dir /path/to/output \
        --prefix object_001 \
        [--seed 42]

Outputs:
    {output_dir}/{prefix}_mesh_vertices.npy
    {output_dir}/{prefix}_mesh_faces.npy
    {output_dir}/{prefix}_gs_points.npy
    {output_dir}/{prefix}_gs_colors.npy
    {output_dir}/{prefix}_pointcloud.ply
    {output_dir}/{prefix}_mesh.obj

Prints "SUCCESS" on last line if everything worked.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="Path to RGB image (BGR, .png)")
    parser.add_argument("--mask", required=True, help="Path to mask (.npy, bool)")
    parser.add_argument("--pointmap", required=True, help="Path to pointmap (.npy, float32 HxWx3)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="recon", help="Output filename prefix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Setup paths ──
    sam3d_root = Path(__file__).resolve().parent
    for p in [str(sam3d_root), str(sam3d_root / "notebook")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    hf_cache = sam3d_root / ".cache" / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    # ── Load inputs ──
    import cv2
    print(f"[Bridge] Loading inputs...")
    rgb_bgr = cv2.imread(args.rgb)
    if rgb_bgr is None:
        print(f"[Bridge] ERROR: Failed to read {args.rgb}")
        sys.exit(1)
    mask = np.load(args.mask).astype(bool)
    pointmap = np.load(args.pointmap).astype(np.float32)
    print(f"[Bridge] RGB: {rgb_bgr.shape}, Mask: {mask.shape} ({mask.sum()} px), Pointmap: {pointmap.shape}")

    # ── Load SAM3D pipeline ──
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hydra.utils import instantiate
    from inference import BLACKLIST_FILTERS, WHITELIST_FILTERS, check_hydra_safety
    from omegaconf import OmegaConf

    config_path = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    print(f"[Bridge] Loading pipeline from {config_path} ...")
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    config.depth_model = None
    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    pipeline = instantiate(config)
    print("[Bridge] Pipeline loaded.")

    # ── Run reconstruction ──
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    # BGR -> RGB -> RGBA (mask as alpha)
    rgb_image = rgb_bgr[:, :, ::-1].copy()
    alpha = mask.astype(np.uint8) * 255
    rgba_image = np.concatenate([rgb_image, alpha[..., None]], axis=-1)
    pointmap_tensor = torch.from_numpy(pointmap).float()

    print("[Bridge] Running reconstruction...")
    output = pipeline.run(
        rgba_image,
        None,
        seed=args.seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=pointmap_tensor,
    )

    # ── Extract results ──
    mesh_result = output["mesh"][0]
    vertices_local = mesh_result.vertices
    faces = mesh_result.faces.detach().cpu().numpy()

    rotation = output["rotation"]
    translation = output["translation"]
    scale = output["scale"]
    transform = compose_transform(
        scale=scale,
        rotation=quaternion_to_matrix(rotation),
        translation=translation,
    )

    vertices_scene = (
        transform.transform_points(vertices_local.unsqueeze(0))[0]
        .detach().cpu().numpy()
    ).astype(np.float32)
    faces = faces.astype(np.int32)

    gs_points_scene = (
        transform.transform_points(output["gs"].get_xyz.unsqueeze(0))[0]
        .detach().cpu().numpy()
    ).astype(np.float32)

    gs_colors = output["gs"].get_features.detach().cpu().numpy()
    gs_colors = np.squeeze(gs_colors)
    gs_colors = gs_colors.reshape(gs_points_scene.shape[0], -1)[:, :3]
    gs_colors = np.clip(gs_colors * 255.0, 0, 255).astype(np.uint8)

    # ── Save outputs ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    # Save raw numpy arrays
    np.save(str(out / f"{prefix}_mesh_vertices.npy"), vertices_scene)
    np.save(str(out / f"{prefix}_mesh_faces.npy"), faces)
    np.save(str(out / f"{prefix}_gs_points.npy"), gs_points_scene)
    np.save(str(out / f"{prefix}_gs_colors.npy"), gs_colors)

    # Save PLY point cloud
    ply_path = str(out / f"{prefix}_pointcloud.ply")
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {gs_points_scene.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(gs_points_scene, gs_colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    # Save OBJ mesh
    obj_path = str(out / f"{prefix}_mesh.obj")
    with open(obj_path, "w") as f:
        for v in vertices_scene:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"[Bridge] Saved: {prefix}_mesh.obj ({len(vertices_scene)} verts, {len(faces)} faces)")
    print(f"[Bridge] Saved: {prefix}_pointcloud.ply ({len(gs_points_scene)} points)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
