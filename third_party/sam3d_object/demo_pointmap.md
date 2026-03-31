# demo_pointmap.py - Point Map to 3D Object Pipeline

## Purpose
Processes a point map (3D point cloud with RGB data) to generate a 3D mesh and Gaussian splatting representation, with optional alignment to reference geometry.

## Input Data
- **Point map**: Zarr file containing `obj_xyz` (H×W×3 point coordinates) and `obj_rgb` (H×W×3 colors)
- **Mask**: PNG image defining valid regions
- **Optional**: Reference point cloud for alignment comparison (`.npy`, `.obj`, `.ply`)

## Pipeline Flow
1. Load point map and RGB from Zarr archive
2. Load and resize mask to match image dimensions
3. Merge mask with RGB to create RGBA input
4. Run SAM 3D Objects pipeline with point map input
5. Transform output from local to scene coordinates using predicted pose (rotation, translation, scale)
6. Export mesh (OBJ) and Gaussian splat points (PLY)
7. Align output to reference geometry using ICP registration
8. Report size ratios and alignment metrics

## Key Functions
- `merge_mask_to_rgba()`: Converts mask to alpha channel
- `rigid_match()`: ICP alignment between source and target point clouds
- `sorted_obb_extent()`: Computes oriented bounding box dimensions
- `save_obj()`, `save_pointcloud_ply()`: Export 3D data

## Output Files
- `mesh_pointmap_scene.obj`: Mesh in scene coordinates
- `pointcloud_pointmap_scene.ply`: Gaussian splat points with colors

## Metrics Reported
- Target vs predicted OBB extents (sorted dimensions)
- Size ratio (predicted/target)
- ICP fitness score
- ICP RMSE

## Configuration
- Uses `checkpoints/hf/pipeline.yaml`
- Disables depth model (not needed for point map input)
- Uses PyTorch3D rendering engine
