import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from research_pipeline.mesh_analyzer import cluster_triangle_threshold


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("Post-processing mesh; keeping up to {} clusters".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = cluster_triangle_threshold(cluster_n_triangles, cluster_to_keep)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def extract_mesh(dataset, pipe, iteration, num_cluster=1):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    kernel_size = dataset.kernel_size
    depth_name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = scene.getTrainCameras()

    # [RESOLUSI AMAN] Mengunci detail di 6 milimeter
    voxel_size = 0.009
    o3d_device = o3d.core.Device("CPU:0")
    
    print(f"[PELACAK] Fase 2: Inisialisasi VoxelBlockGrid dengan resolusi {voxel_size}...")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=20000,
        device=o3d_device,
    )
    print("[PELACAK] VoxelBlockGrid berhasil dibuat!")

    print("\n[PELACAK] Merender dan mengintegrasikan setiap kamera satu per satu...")
    total_cameras = len(viewpoint_cam_list)
    for camera_id, viewpoint_cam in enumerate(viewpoint_cam_list, 1):
        print(f"[PELACAK] Kamera {camera_id}/{total_cameras}: render dan integrasi TSDF...", flush=True)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        color_array = np.ascontiguousarray(
            torch.clamp(render_pkg["render"], min=0, max=1.0)
            .cpu().numpy().transpose(1, 2, 0)
        )
        depth_tensor = render_pkg[depth_name]
        if viewpoint_cam.gt_mask is not None:
            depth_tensor = depth_tensor.clone()
            depth_tensor[viewpoint_cam.gt_mask < 0.5] = 0
        depth_array = np.ascontiguousarray(depth_tensor[0].cpu().numpy())
        depth = o3d.t.geometry.Image(depth_array).to(o3d_device)
        color = o3d.t.geometry.Image(color_array).to(o3d_device)
        intrinsic = np.array([[viewpoint_cam.Fx, 0, viewpoint_cam.Cx], [0, viewpoint_cam.Fy, viewpoint_cam.Cy], [0, 0, 1]], dtype=np.float64)
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        
        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1.0, 7.0)
        vbg.integrate(frustum_block_coords, depth, color, intrinsic, extrinsic, 1.0, 7.0)
        del render_pkg, color_array, depth_tensor, depth_array, depth, color

    # --- DI SINI LETAK PERUBAHAN EDIT KODE BARU KITA ---
    print("\n[PELACAK] Fase 3 Selesai! Mulai mengekstrak Solid Mesh...")
    
    # [OPTIMASI KRUSIAL] Abaikan voxel kosong berbobot rendah agar RAM tidak meledak
    mesh = vbg.extract_triangle_mesh(weight_threshold=3.0)
    
    # [PERBAIKAN BUG] Konversi ke legacy format agar fungsi Open3D bisa menyimpan file PLY
    print("[PELACAK] Menyimpan mesh mentah...")
    legacy_mesh = mesh.to_legacy()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon_raw.ply"), legacy_mesh)
    
    # Pembersihan RAM agresif sebelum masuk algoritma clustering post-process
    del vbg
    del mesh
    torch.cuda.empty_cache()
    
    print("\n[PELACAK] Fase 4: Post-processing mesh...")
    final_mesh = post_process_mesh(legacy_mesh, num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon_post.ply"), final_mesh)
    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        extract_mesh(model.extract(args), pipeline.extract(args), args.iteration, args.num_cluster)
