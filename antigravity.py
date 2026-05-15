"""
Antigravity — Real Estate 3D Intelligence Pipeline
=====================================================
Transforms a single 2D real estate image into an interactive 3D experience.

Pipeline:
  1. Image analysis (scene classification, lighting, materials)
  2. Monocular depth estimation (Apple Depth Pro)
  3. Dense point cloud construction (RGB + depth)
  4. Geometry cleaning and mesh reconstruction
  5. GLB export for web rendering
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image, ImageStat, ImageFilter

# Setup logging FIRST - before any logger calls
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler()
    ]
)
# Force immediate flush for debugging
for handler in logging.root.handlers:
    handler.flush()

logger = logging.getLogger("antigravity")

# Add depth_pro to path
DEPTH_PRO_SRC = Path(__file__).parent / "ml-depth-pro" / "src"
logger.info(f"  Adding ml-depth-pro path: {DEPTH_PRO_SRC}")
logger.info(f"  Path exists: {DEPTH_PRO_SRC.exists()}")

sys.path.insert(0, str(DEPTH_PRO_SRC))
logger.info("  Importing depth_pro...")

import depth_pro
logger.info(f"  depth_pro imported: {depth_pro}")


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SceneAnalysis:
    """Result of image analysis."""
    scene_type: str          # "exterior_villa", "exterior_building", "interior_room", "interior_hallway"
    category: str            # "exterior" or "interior"
    lighting: str            # "bright", "moderate", "dark"
    brightness: float        # 0-255
    contrast: float
    dominant_colors: list    # top 3 hex colors
    estimated_materials: list
    width: int
    height: int


@dataclass
class PipelineResult:
    """Final pipeline output."""
    scene_type: str
    depth_range_meters: list  # [min, max]
    point_count: int
    mesh_faces: int
    processing_time_sec: float
    glb_path: str
    ply_path: str
    depth_map_path: str


# ─────────────────────────────────────────────────────────────────────
# Stage 1: Image Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_image(image_path: str) -> SceneAnalysis:
    """Classify scene type, detect lighting, estimate materials."""
    logger.info("Stage 1: Analyzing image...")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    stat = ImageStat.Stat(img)
    
    # Brightness & contrast
    brightness = stat.mean[0] * 0.299 + stat.mean[1] * 0.587 + stat.mean[2] * 0.114
    contrast = (stat.stddev[0] + stat.stddev[1] + stat.stddev[2]) / 3.0
    
    if brightness > 170:
        lighting = "bright"
    elif brightness > 85:
        lighting = "moderate"
    else:
        lighting = "dark"
    
    # Dominant color extraction via quantization
    small = img.resize((64, 64))
    quantized = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:24]  # 8 colors × 3 channels
    dominant_colors = []
    for i in range(0, min(len(palette), 9), 3):
        r, g, b = palette[i], palette[i+1], palette[i+2]
        dominant_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    dominant_colors = dominant_colors[:3]
    
    # Scene classification heuristics
    # Analyze spatial frequency distribution (edges in different regions)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)
    
    # Split image into regions
    h3, w3 = h // 3, w // 3
    top_edge = edge_arr[:h3, :].mean()
    mid_edge = edge_arr[h3:2*h3, :].mean()
    bot_edge = edge_arr[2*h3:, :].mean()
    left_edge = edge_arr[:, :w3].mean()
    right_edge = edge_arr[:, 2*w3:].mean()
    
    # Color analysis for sky detection
    top_region = np.array(img)[:h//4, :, :]
    top_blue_ratio = top_region[:,:,2].mean() / (top_region.mean() + 1e-6)
    top_brightness = top_region.mean()
    
    # Interior vs Exterior classification
    has_sky = top_blue_ratio > 1.15 or top_brightness > 180
    edge_uniformity = np.std([top_edge, mid_edge, bot_edge])
    
    # Interior scenes tend to have more uniform edge distribution and 
    # higher edge density on borders (walls, corners)
    border_edge_ratio = (left_edge + right_edge) / (mid_edge + 1e-6)
    
    if has_sky and top_edge < mid_edge:
        category = "exterior"
        # Distinguish villa vs building by aspect ratio and edge patterns
        if w / h > 1.3 and bot_edge > mid_edge:
            scene_type = "exterior_villa"
        else:
            scene_type = "exterior_building"
    else:
        category = "interior"
        if border_edge_ratio > 1.5:
            scene_type = "interior_hallway"
        else:
            scene_type = "interior_room"
    
    # Material estimation based on color and texture
    materials = []
    avg_color = np.array(img).mean(axis=(0,1))
    
    if category == "exterior":
        if avg_color[0] > 180 and avg_color[1] > 170 and avg_color[2] > 160:
            materials.append("white_stucco")
        if contrast > 50:
            materials.append("stone")
        materials.append("glass")
        if brightness < 140:
            materials.append("dark_cladding")
        else:
            materials.append("concrete")
    else:
        if brightness > 160:
            materials.append("white_walls")
        if contrast < 40:
            materials.append("smooth_plaster")
        else:
            materials.append("textured_wall")
        materials.append("flooring")
        if avg_color[0] > avg_color[2] + 20:
            materials.append("wood")
    
    result = SceneAnalysis(
        scene_type=scene_type,
        category=category,
        lighting=lighting,
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        dominant_colors=dominant_colors,
        estimated_materials=materials,
        width=w,
        height=h,
    )
    
    logger.info(f"  Scene: {scene_type} | Lighting: {lighting} | "
                f"Size: {w}x{h} | Materials: {materials}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Depth Estimation
# ─────────────────────────────────────────────────────────────────────

def estimate_depth(
    image_path: str,
    checkpoint_path: str = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Run Apple Depth Pro on the input image.
    
    Returns:
        depth_map: HxW numpy array of metric depth in meters
        focal_length_px: estimated focal length in pixels
        rgb_image: HxW×3 numpy array (for point cloud coloring)
    """
    logger.info("Stage 2: Running Depth Pro inference...")
    
    if checkpoint_path is None:
        checkpoint_path = str(Path(__file__).parent / "checkpoints" / "depth_pro.pt")
    
    logger.info(f"  Checkpoint path: {checkpoint_path}")
    logger.info(f"  Checkpoint exists: {Path(checkpoint_path).exists()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    
    # Load model
    logger.info("  Creating DepthProConfig...")
    config = depth_pro.depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=checkpoint_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    logger.info("  Config created, now loading model...")
    
    logger.info("  Calling create_model_and_transforms()...")
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=torch.float16 if device.type == "cuda" else torch.float32,
    )
    logger.info("  Model loaded successfully!")
    model.eval()
    
    # Load image
    logger.info("  Loading image with depth_pro.load_rgb()...")
    rgb, _, f_px = depth_pro.load_rgb(image_path)
    logger.info(f"  RGB loaded: {type(rgb)}, f_px: {f_px}")
    
    logger.info("  Applying transform...")
    image_tensor = transform(rgb)
    logger.info(f"  image_tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
    
    # Inference - wrap in try/except to catch exact failure point
    logger.info("  ABOUT TO CALL model.infer()...")
    try:
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px)
        logger.info("  model.infer() COMPLETED!")
    except Exception as e:
        logger.error(f"  FAILED during model.infer(): {e}")
        raise
    
    logger.info("  Extracting depth and focal from prediction...")
    logger.info(f"  Prediction keys: {list(prediction.keys())}")
    
    # Fix 2: Force 2D depth shape with defensive handling
    depth = prediction["depth"]
    if hasattr(depth, 'squeeze'):
        depth = depth.squeeze()
    depth = depth.cpu().numpy()
    if depth.ndim != 2:
        logger.warning(f"  WARNING: Unexpected depth shape {depth.shape}, squeezing...")
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")
    
    # Fix 3: Safe focal length key access with fallback
    if "focallength_px" in prediction:
        focal = prediction["focallength_px"].cpu().item()
    elif "f_px" in prediction:
        focal = prediction["f_px"].cpu().item()
    else:
        logger.warning("  WARNING: No focal key found, using default 1.0")
        focal = 1.0
    
    logger.info(f"  Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    logger.info(f"  Focal length: {focal:.1f}px")
    logger.info(f"  Depth map shape: {depth.shape}")
    
    # Free GPU memory
    del model, image_tensor, prediction
    # Fix 1: Convert expression to statement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return depth, focal, rgb


# ─────────────────────────────────────────────────────────────────────
# Stage 3: Point Cloud Construction
# ─────────────────────────────────────────────────────────────────────

def build_point_cloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    focal_length_px: float,
    scene: SceneAnalysis,
) -> o3d.geometry.PointCloud:
    """
    Back-project depth map into a dense 3D point cloud with RGB colors.
    Uses pinhole camera model.
    """
    logger.info("Stage 3: Constructing point cloud...")
    
    H, W = depth_map.shape
    
    # Ensure RGB matches depth dimensions
    if rgb_image.shape[0] != H or rgb_image.shape[1] != W:
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil = rgb_pil.resize((W, H), Image.LANCZOS)
        rgb_image = np.array(rgb_pil)
    
    # Build pixel coordinate grids
    cx, cy = W / 2.0, H / 2.0
    fx = fy = focal_length_px
    
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Back-project to 3D
    Z = depth_map.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack into Nx3 points
    points = np.stack([X, -Y, -Z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0
    
    # Filter out invalid points
    valid_depth = Z.reshape(-1)
    mask = (valid_depth > 0.01) & (valid_depth < 200.0) & np.isfinite(valid_depth)
    points = points[mask]
    colors = colors[mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    logger.info(f"  Raw point count: {len(pcd.points):,}")
    return pcd


# ─────────────────────────────────────────────────────────────────────
# Stage 4: Geometry Cleaning & Mesh Reconstruction
# ─────────────────────────────────────────────────────────────────────

def clean_and_reconstruct(
    pcd: o3d.geometry.PointCloud,
    scene: SceneAnalysis,
    output_dir: Path,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Clean point cloud and reconstruct a mesh.
    Adapts parameters based on scene type.
    """
    logger.info("Stage 4: Cleaning geometry and reconstructing mesh...")
    
    # Adaptive parameters based on scene type
    if scene.category == "exterior":
        voxel_size = 0.04        # coarser for large-scale scenes
        nb_neighbors = 30
        std_ratio = 1.5
        poisson_depth = 10
    else:
        voxel_size = 0.02        # finer for interior detail
        nb_neighbors = 40
        std_ratio = 1.8
        poisson_depth = 11
    
    # Adjust for dark scenes (noisier depth)
    if scene.lighting == "dark":
        std_ratio *= 0.8         # more aggressive outlier removal
        nb_neighbors = 50
    
    # 1. Voxel downsampling
    logger.info(f"  Voxel downsampling (size={voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    logger.info(f"  Points after downsampling: {len(pcd_down.points):,}")
    
    # 2. Statistical outlier removal
    logger.info(f"  Outlier removal (neighbors={nb_neighbors}, std={std_ratio})...")
    pcd_clean, mask = pcd_down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    logger.info(f"  Points after cleaning: {len(pcd_clean.points):,}")
    
    # 3. Estimate normals
    logger.info("  Estimating normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4, max_nn=30
        )
    )
    pcd_clean.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    # 4. Mesh reconstruction — Poisson Surface Reconstruction
    logger.info(f"  Poisson reconstruction (depth={poisson_depth})...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=poisson_depth, linear_fit=True
        )
        
        # Remove low-density vertices (trim artifacts at boundaries)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.02)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"  Mesh: {len(mesh.vertices):,} vertices, "
                     f"{len(mesh.triangles):,} faces")
    except Exception as e:
        logger.warning(f"  Poisson failed: {e}. Falling back to Ball-Pivoting...")
        # Ball-pivoting fallback
        radii = [voxel_size * 2, voxel_size * 4, voxel_size * 8]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_clean, o3d.utility.DoubleVector(radii)
        )
        logger.info(f"  Mesh (BPA): {len(mesh.vertices):,} vertices, "
                     f"{len(mesh.triangles):,} faces")
    
    # 5. Post-process mesh
    mesh.compute_vertex_normals()
    
    # Transfer colors from point cloud to mesh vertices
    if not mesh.has_vertex_colors():
        # Fix 8: Vectorized KNN color transfer using scipy cKDTree
        import scipy.spatial
        pcd_points = np.asarray(pcd_clean.points)
        pcd_colors = np.asarray(pcd_clean.colors)
        mesh_verts = np.asarray(mesh.vertices)
        
        tree = scipy.spatial.cKDTree(pcd_points)
        _, indices = tree.query(mesh_verts, k=1)
        mesh_colors = pcd_colors[indices]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    # Save PLY
    ply_path = output_dir / "model.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    logger.info(f"  Saved point cloud: {ply_path}")
    
    return mesh, pcd_clean


# ─────────────────────────────────────────────────────────────────────
# Stage 5: GLB Export
# ─────────────────────────────────────────────────────────────────────

def export_glb(
    mesh: o3d.geometry.TriangleMesh,
    output_dir: Path,
) -> str:
    """Export Open3D mesh to GLB via trimesh."""
    logger.info("Stage 5: Exporting GLB...")
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Vertex colors
    if mesh.has_vertex_colors():
        colors_float = np.asarray(mesh.vertex_colors)
        # Convert to RGBA uint8
        colors_uint8 = (np.clip(colors_float, 0, 1) * 255).astype(np.uint8)
        colors_rgba = np.hstack([
            colors_uint8,
            np.full((len(colors_uint8), 1), 255, dtype=np.uint8)
        ])
    else:
        colors_rgba = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
    
    # Create trimesh
    # Fix 4: process=False to preserve vertex color mapping
    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors_rgba,
        process=False,
    )
    
    # Center and normalize scale for web viewing
    tri_mesh.apply_translation(-tri_mesh.centroid)
    extent = tri_mesh.bounding_box.extents.max()
    if extent > 0:
        scale = 5.0 / extent  # normalize to ~5 units
        tri_mesh.apply_scale(scale)
    
    # Export as GLB
    glb_path = output_dir / "model.glb"
    tri_mesh.export(str(glb_path), file_type="glb")
    
    file_size_mb = glb_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Exported: {glb_path} ({file_size_mb:.1f} MB)")
    
    return str(glb_path)


# ─────────────────────────────────────────────────────────────────────
# Save depth map as visualization
# ─────────────────────────────────────────────────────────────────────

def save_depth_visualization(depth_map: np.ndarray, output_dir: Path) -> str:
    """Save a colorized depth map for display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    depth_vis = depth_map.copy()
    d_min, d_max = np.percentile(depth_vis, [2, 98])
    depth_vis = np.clip(depth_vis, d_min, d_max)
    depth_normalized = (depth_vis - d_min) / (d_max - d_min + 1e-6)
    
    cmap = plt.get_cmap("magma")
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    
    out_path = output_dir / "depth_map.png"
    Image.fromarray(depth_colored).save(str(out_path))
    logger.info(f"  Saved depth visualization: {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = None) -> PipelineResult:
    """
    Execute the full Antigravity pipeline.
    
    Args:
        image_path: Path to the input 2D image.
        output_dir: Directory for outputs (created if needed).
    
    Returns:
        PipelineResult with all metadata and paths.
    """
    start_time = time.time()
    image_path = str(Path(image_path).resolve())
    
    if output_dir is None:
        output_dir = str(Path(image_path).parent / "output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"{'='*60}")
    logger.info(f"ANTIGRAVITY — Real Estate 3D Intelligence Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Input: {image_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Stage 1: Analyze
    scene = analyze_image(image_path)
    
    # Stage 2: Depth estimation
    depth_map, focal_length, rgb = estimate_depth(image_path)
    depth_path = save_depth_visualization(depth_map, output_dir)
    
    # Stage 3: Point cloud
    pcd = build_point_cloud(depth_map, rgb, focal_length, scene)
    
    # Stage 4: Clean & reconstruct
    mesh, pcd_clean = clean_and_reconstruct(pcd, scene, output_dir)
    
    # Stage 5: Export GLB
    glb_path = export_glb(mesh, output_dir)
    
    elapsed = time.time() - start_time
    
    result = PipelineResult(
        scene_type=scene.scene_type,
        depth_range_meters=[round(float(depth_map.min()), 2), round(float(depth_map.max()), 2)],
        point_count=len(pcd_clean.points),
        mesh_faces=len(mesh.triangles),
        processing_time_sec=round(elapsed, 1),
        # Fix 6: Use relative filenames only (not absolute paths)
        glb_path=Path(glb_path).name,
        ply_path="model.ply",
        depth_map_path="depth_map.png",
    )
    
    # Save JSON summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(f"  Summary: {summary_path}")
    
    logger.info(f"{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Scene: {result.scene_type}")
    logger.info(f"  Depth: {result.depth_range_meters[0]}m – {result.depth_range_meters[1]}m")
    logger.info(f"  Points: {result.point_count:,}")
    logger.info(f"  Faces: {result.mesh_faces:,}")
    logger.info(f"  GLB: {result.glb_path}")
    logger.info(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py antigravity.py <image_path> [output_dir]")
        sys.exit(1)
    
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(img, out)
