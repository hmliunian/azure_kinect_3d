import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image as PILImage
from pytorch3d.transforms import quaternion_to_matrix

# Keep Hugging Face writes inside the repo to avoid permission issues.
REPO_ROOT = Path(__file__).resolve().parent
HF_CACHE = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))

# Import notebook helpers from the repo.
sys.path.append("notebook")
from inference import BLACKLIST_FILTERS, WHITELIST_FILTERS, check_hydra_safety  # noqa: E402
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform  # noqa: E402


def merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    return np.concatenate([image[..., :3], alpha[..., None]], axis=-1)


def save_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for x, y, z in points:
                f.write(f"{x} {y} {z}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def load_obj_vertices(path: str) -> np.ndarray:
    vertices = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {path}")
    return np.asarray(vertices, dtype=np.float32)


def load_part_pointcloud(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
        if points.ndim == 3 and points.shape[-1] == 3:
            points = points.reshape(-1, 3)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Unsupported NPY point cloud shape: {points.shape}")
        return np.asarray(points[:, :3], dtype=np.float32)
    if suffix == ".obj":
        return load_obj_vertices(path)

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Failed to load points from: {path}")
    return points


def make_o3d_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def sorted_obb_extent(points: np.ndarray) -> np.ndarray:
    pcd = make_o3d_pointcloud(points)
    extent = np.asarray(pcd.get_oriented_bounding_box().extent, dtype=np.float32)
    return np.sort(extent)


def estimate_voxel_size(source_points: np.ndarray, target_points: np.ndarray) -> float:
    extent = max(
        np.ptp(source_points, axis=0).max(initial=0.0),
        np.ptp(target_points, axis=0).max(initial=0.0),
    )
    return max(float(extent) / 50.0, 1e-3)


def rigid_match(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, object]:
    source = make_o3d_pointcloud(source_points)
    target = make_o3d_pointcloud(target_points)

    voxel_size = estimate_voxel_size(source_points, target_points)
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    source_center = np.asarray(source_down.get_center())
    target_center = np.asarray(target_down.get_center())
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = target_center - source_center

    reg = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        voxel_size * 3.0,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    source.transform(reg.transformation)
    return np.asarray(source.points, dtype=np.float32), reg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part-pointcloud",
        type=str,
        default=None,
        help="Optional input part point cloud to compare against the predicted scene point cloud.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tag = "hf"
    config_path = REPO_ROOT / "checkpoints" / tag / "pipeline.yaml"
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    # Point-map demos do not need the MoGe depth model.
    config.depth_model = None
    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    pipeline = instantiate(config)

    zarr_path = "/home/xuran/dataset2/expo/pointcloud.zarr"
    z = zarr.open(zarr_path, mode="r")

    pointmap = np.array(z["obj_xyz"], dtype=np.float32)  # (H, W, 3)
    image = np.array(z["obj_rgb"])
    if image.dtype == np.float32:
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = (image * 255).clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    mask_img = PILImage.open("/home/xuran/dataset2/expo/mask.png")
    mask_img = mask_img.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
    mask = np.array(mask_img)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)
    valid_mask = mask.astype(bool) & np.isfinite(pointmap).all(axis=-1)

    rgba_image = merge_mask_to_rgba(image, mask)
    pointmap_tensor = torch.from_numpy(pointmap).float()
    output = pipeline.run(
        rgba_image,
        None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=pointmap_tensor,
    )

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
        .detach()
        .cpu()
        .numpy()
    )

    save_obj("mesh_pointmap_scene.obj", vertices_scene, faces)

    gs_points_scene = (
        transform.transform_points(output["gs"].get_xyz.unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
    )
    gs_colors = (
        output["gs"].get_features
        .detach()
        .cpu()
        .numpy()
    )
    gs_colors = np.squeeze(gs_colors)
    gs_colors = gs_colors.reshape(gs_points_scene.shape[0], -1)[:, :3]
    gs_colors = np.clip(gs_colors * 255.0, 0, 255).astype(np.uint8)
    save_pointcloud_ply("pointcloud_pointmap_scene.ply", gs_points_scene, gs_colors)

    if args.part_pointcloud is not None:
        target_points = load_part_pointcloud(args.part_pointcloud)
        target_label = args.part_pointcloud
    else:
        target_points = pointmap[valid_mask]
        target_label = "masked obj_xyz from point map"

    matched_output_points, reg = rigid_match(gs_points_scene, target_points)
    target_extent = sorted_obb_extent(target_points)
    output_extent = sorted_obb_extent(matched_output_points)
    ratio = output_extent / np.clip(target_extent, 1e-8, None)

    print(f"Target point cloud: {target_label}")
    print(f"Target OBB extent (sorted): {target_extent}")
    print(f"Predicted OBB extent (sorted): {output_extent}")
    print(f"Predicted/target size ratio: {ratio}")
    print(f"ICP fitness: {reg.fitness:.6f}")
    print(f"ICP RMSE: {reg.inlier_rmse:.6f}")


if __name__ == "__main__":
    main()
