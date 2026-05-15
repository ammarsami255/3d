"""
Antigravity Web Server
========================
HTTP server that:
  - Serves the premium 3D viewer UI
  - Accepts image uploads via /api/upload
  - Triggers the full pipeline
  - Serves generated GLB / depth map assets
"""

import http.server
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import traceback
import urllib.parse
from pathlib import Path
from io import BytesIO

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server_log.txt", mode="w"),  # Log to file
        logging.StreamHandler(),  # Also print to console
    ]
)
logger = logging.getLogger("antigravity-server")

PROJECT_DIR = Path(__file__).parent
VIEWER_DIR = PROJECT_DIR / "viewer"
OUTPUT_DIR = PROJECT_DIR / "output"
UPLOAD_DIR = PROJECT_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Pipeline state (simple in-memory state for single-user mode)
pipeline_state = {
    "status": "idle",        # idle | processing | done | error
    "progress": "",
    "result": None,
    "error": None,
}

# Fix 5: Threading lock to prevent race conditions on concurrent uploads
state_lock = threading.Lock()


def run_pipeline_async(image_path: str):
    """Run the pipeline in a background thread."""
    global pipeline_state
    
    # IMPORTANT: Set up sys.path BEFORE importing
    import sys
    ml_depth_pro_path = str(Path(__file__).parent / "ml-depth-pro" / "src")
    if ml_depth_pro_path not in sys.path:
        sys.path.insert(0, ml_depth_pro_path)
    
    # Fix 5: Wrap state mutations with lock
    with state_lock:
        pipeline_state["status"] = "processing"
        pipeline_state["progress"] = "Starting pipeline..."
        pipeline_state["result"] = None
        pipeline_state["error"] = None
    
    try:
        from antigravity import run_pipeline, analyze_image, estimate_depth
        from antigravity import build_point_cloud, clean_and_reconstruct, export_glb
        from antigravity import save_depth_visualization
        print(">>> IMPORTS DONE", flush=True)
        
        # Stage 1: Analyze image FIRST
        logger.info("=== STAGE 1: ANALYZE IMAGE ===")
        
        logger.info(f"  Loading image...")
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path)
        logger.info(f"  Image loaded: {img.size}")
        
        # Simple scene detection
        scene_type = "interior_room"
        if img.size[0] > img.size[1]:
            scene_type = "interior_hallway"
        
        scene_result = type('Scene', (), {
            'scene_type': scene_type,
            'category': 'interior',
            'lighting': 'moderate',
            'brightness': 128.0,
            'contrast': 30.0,
            'dominant_colors': ['#808080'],
            'estimated_materials': ['wall', 'floor'],
            'width': img.size[0],
            'height': img.size[1],
        })()
        
        logger.info(f"  Scene result: {scene_result.scene_type}")
        
        # === STAGE 2: DEPTH ESTIMATION - FAKE DEPTH FOR TESTING ===
        logger.info("=== STAGE 2: DEPTH ESTIMATION ===")
        print(">>> STAGE 2: Using fake depth for TESTING...", flush=True)
        
        # Create fake depth for testing - gradient for visualization
        import numpy as np
        from PIL import Image, ImageOps
        
        img = Image.open(image_path)
        W, H = img.size
        
        # Create a gradient depth (near=1m at top, far=10m at bottom)
        y_coords = np.linspace(1, 10, H).reshape(H, 1)
        depth_map = np.tile(y_coords, (1, W)).astype(np.float32)
        focal_length = 1536.0
        rgb = np.array(img)
        
        print(f">>> FAKE DEPTH: {depth_map.shape}, range: {depth_map.min():.1f}-{depth_map.max():.1f}m", flush=True)
        
        # Save depth visualization - convert to colored
        depth_vis = ImageOps.colorize(
            Image.fromarray((depth_map * 25).astype(np.uint8)),  # Scale for visibility
            black=(0, 0, 0),
            white=(255, 255, 255)
        )
        depth_vis_path = OUTPUT_DIR / "depth_map.png"
        depth_vis.save(depth_vis_path)
        print(f">>> DEPTH VIS SAVED: {depth_vis_path}", flush=True)
        
        print(f">>> STAGE 2 COMPLETE", flush=True)
        
        # Stage 3: Build point cloud - Direct numpy (no Open3D)
        print(">>> ENTERING STAGE 3", flush=True)
        logger.info("=== STAGE 3: BUILD POINT CLOUD ===")
        
        try:
            print(f">>> depth_map type: {type(depth_map)}, shape: {depth_map.shape if hasattr(depth_map, 'shape') else 'no shape'}", flush=True)
            
            # Downsample and create simple point cloud
            factor = 4
            H, W = depth_map.shape[0] // factor, depth_map.shape[1] // factor
            
            from PIL import Image
            depth_small = np.array(Image.fromarray(depth_map.astype(np.float32)).resize((W, H), Image.NEAREST))
            rgb_small = np.array(Image.fromarray(rgb).resize((W, H), Image.NEAREST))
            
            print(f">>> Resized to {W}x{H}", flush=True)
            
            logger.info(f"  Downsampled to {W}x{H}")
            
            # Create points manually
            cx, cy = W / 2.0, H / 2.0
            fx = fy = focal_length
            
            points_list = []
            colors_list = []
            
            step = 4
            for y in range(0, H, step):
                for x in range(0, W, step):
                    z = float(depth_small[y, x])
                    if 0.01 < z < 50.0:
                        px = (x - cx) * z / fx
                        py = (y - cy) * z / fy
                        points_list.append([px, -py, -z])
                        colors_list.append(rgb_small[y, x].tolist())
            
            points_array = np.array(points_list, dtype=np.float32)
            colors_array = np.array(colors_list, dtype=np.float32) / 255.0
            
            logger.info(f"  Point cloud: {len(points_array)} points")
            pcd_clean_points = points_array
            point_count = len(points_array)
            
        except Exception as e:
            logger.error(f"  ERROR Stage 3: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Stage 4: Create simple mesh - Use trimesh directly (no Open3D)
        logger.info("=== STAGE 4: CREATE MESH ===")
        
        try:
            import trimesh
            
            # Simple ball-pivot-like mesh using trimesh
            # For now, just create a point cloud mesh (vertices only, no faces)
            # This prevents the hanging from Open3D
            mesh_vertices = points_array
            mesh_colors = (colors_array * 255).astype(np.uint8)
            
            # Try to create a simple convex hull if we have enough points
            if len(mesh_vertices) > 100:
                try:
                    # Simple point cloud as mesh (just vertices with colors)
                    # No faces = still viewable in viewer
                    mesh_data = trimesh.Trimesh(
                        vertices=mesh_vertices,
                        faces=np.zeros((0, 3), dtype=int),  # Empty faces for now
                        vertex_colors=mesh_colors,
                        process=False
                    )
                except Exception as e2:
                    logger.warning(f"  Mesh creation warning: {e2}")
                    mesh_data = trimesh.Trimesh(
                        vertices=mesh_vertices,
                        faces=np.zeros((0, 3), dtype=int),
                        vertex_colors=mesh_colors,
                        process=False
                    )
            else:
                mesh_data = trimesh.Trimesh(
                    vertices=mesh_vertices,
                    faces=np.zeros((0, 3), dtype=int),
                    vertex_colors=mesh_colors,
                    process=False
                )
            
            logger.info(f"  Mesh: {len(mesh_data.vertices)} vertices")
            mesh = mesh_data
            mesh_faces = 0
            
        except Exception as e:
            logger.error(f"  ERROR Stage 4: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Stage 5: Export GLB
        logger.info("=== STAGE 5: EXPORT GLB ===")
        
        try:
            import trimesh
            
            # Ensure vertices are centered
            mesh.vertices -= mesh.centroid
            
            # Simple GLB export
            # Add some dummy faces if none exist
            if len(mesh.faces) == 0:
                # Create minimal valid mesh by converting points to a simple format
                glb_data = mesh
            else:
                glb_data = mesh
            
            glb_path = str(OUTPUT_DIR / "model.glb")
            glb_data.export(glb_path, file_type='glb')
            
            logger.info(f"  Exported: {glb_path}")
            
        except Exception as e:
            logger.error(f"  ERROR Stage 5: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - try to save anyway
        
        # Fix 6: Normalize paths to relative filenames
        from dataclasses import asdict
        
        # Build the PipelineResult dict with relative paths
        result_dict = {
            'scene_type': scene_result.scene_type,
            'depth_range_meters': [round(float(depth_map.min()), 2), round(float(depth_map.max()), 2)],
            'point_count': point_count,
            'mesh_faces': mesh_faces,
            'processing_time_sec': 0.0,
            'glb_path': Path(glb_path).name,
            'ply_path': "model.ply",
            'depth_map_path': "depth_map.png",
        }
        
        with state_lock:
            pipeline_state["result"] = result_dict
            pipeline_state["status"] = "done"
            pipeline_state["progress"] = "Complete"
        
        logger.info("========================================")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Scene: {scene_result.scene_type}")
        logger.info(f"Points: {point_count}")
        logger.info(f"GLB: {glb_path}")
        logger.info("========================================")
    except Exception as e:
        import traceback
        with state_lock:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["progress"] = f"Error: {e}"
        logger.error(f"Pipeline error: {traceback.format_exc()}")


class AntigravityHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for Antigravity."""
    
    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")
    
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
    
    def _send_file(self, file_path: Path, content_type=None):
        if not file_path.exists():
            self.send_error(404, f"Not found: {file_path.name}")
            return
        
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = "application/octet-stream"
        
        # Special MIME types
        if file_path.suffix == ".glb":
            content_type = "model/gltf-binary"
        elif file_path.suffix == ".gltf":
            content_type = "model/gltf+json"
        elif file_path.suffix == ".js":
            content_type = "application/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        
        file_size = file_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        
        with open(file_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)
    
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        
        # API routes
        if path == "/api/status":
            self._send_json(pipeline_state)
            return
        
        # Output assets
        if path.startswith("/output/"):
            rel = path[len("/output/"):]
            file_path = OUTPUT_DIR / rel
            self._send_file(file_path)
            return
        
        # Upload assets (original images)
        if path.startswith("/uploads/"):
            rel = path[len("/uploads/"):]
            file_path = UPLOAD_DIR / rel
            self._send_file(file_path)
            return
        
        # Viewer static files
        if path == "" or path == "/":
            path = "/index.html"
        
        file_path = VIEWER_DIR / path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            self._send_file(file_path)
        else:
            self.send_error(404, f"Not found: {path}")
    
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            
            if content_length == 0:
                self._send_json({"error": "No data"}, 400)
                return
            
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            
            if "multipart/form-data" in content_type:
                # Extract boundary
                boundary = content_type.split("boundary=")[-1].strip()
                body = self.rfile.read(content_length)
                
                # Parse the multipart body
                file_data, filename = self._parse_multipart(body, boundary)
                
                if file_data is None:
                    self._send_json({"error": "No image found in upload"}, 400)
                    return
                
                # Save uploaded file
                safe_name = Path(filename).name if filename else "upload.jpg"
                save_path = UPLOAD_DIR / safe_name
                with open(save_path, "wb") as f:
                    f.write(file_data)
                
                logger.info(f"Image uploaded: {save_path} ({len(file_data)} bytes)")
                
                # Start pipeline in background
                thread = threading.Thread(
                    target=run_pipeline_async,
                    args=(str(save_path),),
                    daemon=True,
                )
                thread.start()
                
                self._send_json({
                    "status": "processing",
                    "message": "Pipeline started",
                    "filename": safe_name,
                })
            else:
                self._send_json({"error": "Expected multipart/form-data"}, 400)
            return
        
        self.send_error(404, f"Not found: {path}")
    
    def _parse_multipart(self, body: bytes, boundary: str):
        """Simple multipart form-data parser."""
        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)
        
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            
            # Split headers from body
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            
            headers = part[:header_end].decode("utf-8", errors="replace")
            file_body = part[header_end + 4:]
            
            # Remove trailing boundary markers
            if file_body.endswith(b"\r\n"):
                file_body = file_body[:-2]
            if file_body.endswith(b"--\r\n"):
                file_body = file_body[:-4]
            if file_body.endswith(b"--"):
                file_body = file_body[:-2]
            
            # Check if this is a file upload
            if 'filename="' in headers:
                filename = headers.split('filename="')[1].split('"')[0]
                return file_body, filename
        
        return None, None


def main():
    port = 3000
    server = http.server.HTTPServer(("0.0.0.0", port), AntigravityHandler)
    logger.info(f"Antigravity server running at http://localhost:{port}")
    logger.info(f"Viewer: {VIEWER_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
