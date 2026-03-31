import numpy as np
import zarr

# 加载原始点云
z = zarr.open('/home/xuran/dataset2/expo/pointcloud.zarr', mode='r')
original_xyz = np.array(z['obj_xyz'])
original_rgb = np.array(z['obj_rgb'])

# 提取有效点
valid_mask = np.abs(original_xyz).sum(axis=-1) > 0
original_points = original_xyz[valid_mask]
original_colors = original_rgb[valid_mask]

# 转换颜色到 0-255
if original_colors.max() <= 1.0:
    original_colors = (original_colors * 255).astype(np.uint8)
else:
    original_colors = original_colors.astype(np.uint8)

print(f"原始点云: {len(original_points)} 个点")

# 加载生成的点云 (使用 open3d)
import open3d as o3d

pcd_generated = o3d.io.read_point_cloud('splat_pointmap.ply')
generated_points = np.asarray(pcd_generated.points)

# 检查是否有颜色
if pcd_generated.has_colors():
    generated_colors = (np.asarray(pcd_generated.colors) * 255).astype(np.uint8)
else:
    # 使用默认颜色（红色）
    generated_colors = np.full((len(generated_points), 3), [255, 0, 0], dtype=np.uint8)

print(f"生成点云: {len(generated_points)} 个点")

# 合并点云
all_points = np.vstack([original_points, generated_points])
all_colors = np.vstack([original_colors, generated_colors])

# 保存为 PLY
with open('combined_pointcloud.ply', 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(all_points)}\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')

    for i in range(len(all_points)):
        f.write(f'{all_points[i,0]} {all_points[i,1]} {all_points[i,2]} ')
        f.write(f'{all_colors[i,0]} {all_colors[i,1]} {all_colors[i,2]}\n')

print(f"合并点云已保存: combined_pointcloud.ply ({len(all_points)} 个点)")
