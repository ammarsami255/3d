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
        
        # Stage 1: Analyze image FIRST
        logger.info("=== STAGE 1: ANALYZE IMAGE ===")
        with state_lock:
            pipeline_state["progress"] = "Analyzing image (stage 1/5)..."
        
        from pathlib import Path
        image_path_obj = Path(image_path)
        logger.info(f"  Calling analyze_image({image_path})...")
        scene = analyze_image(str(image_path_obj))
        logger.info(f"  Scene result: {scene.scene_type}")
        
        # Stage 2: Depth estimation
        logger.info("=== STAGE 2: DEPTH ESTIMATION ===")
        with state_lock:
            pipeline_state["progress"] = "Running depth estimation (stage 2/5)..."
        
        logger.info(f"  Calling estimate_depth({image_path})...")
        depth_map, focal_length, rgb = estimate_depth(str(image_path_obj))
        logger.info(f"  Depth result: shape {depth_map.shape}")
        
        # Save depth visualization
        depth_vis_path = save_depth_visualization(depth_map, OUTPUT_DIR)
        
        # Stage 3: Build point cloud
        logger.info("=== STAGE 3: BUILD POINT CLOUD ===")
        with state_lock:
            pipeline_state["progress"] = "Building point cloud (stage 3/5)..."
        
        try:
            logger.info(f"  Calling build_point_cloud() with depth_map shape {depth_map.shape}...")
            pcd = build_point_cloud(depth_map, rgb, focal_length, scene)
            logger.info(f"  Point cloud created: {len(pcd.points)} points")
        except Exception as e:
            logger.error(f"  ERROR in build_point_cloud: {e}")
            raise
        
        # Stage 4: Clean and reconstruct mesh
        logger.info("=== STAGE 4: CLEAN & RECONSTRUCT ===")
        with state_lock:
            pipeline_state["progress"] = "Reconstructing mesh (stage 4/5)..."
        
        try:
            logger.info(f"  Calling clean_and_reconstruct()...")
            mesh, pcd_clean = clean_and_reconstruct(pcd, scene, OUTPUT_DIR)
            logger.info(f"  Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        except Exception as e:
            logger.error(f"  ERROR in clean_and_reconstruct: {e}")
            raise
        
        # Stage 5: Export GLB
        logger.info("=== STAGE 5: EXPORT GLB ===")
        with state_lock:
            pipeline_state["progress"] = "Exporting GLB (stage 5/5)..."
        
        logger.info(f"  Calling export_glb()...")
        glb_path = export_glb(mesh, OUTPUT_DIR)
        
        # Fix 6: Normalize paths to relative filenames
        from dataclasses import asdict
        
        # Build the PipelineResult dict with relative paths
        result_dict = {
            'scene_type': scene.scene_type,
            'depth_range_meters': [round(float(depth_map.min()), 2), round(float(depth_map.max()), 2)],
            'point_count': len(pcd_clean.points),
            'mesh_faces': len(mesh.triangles),
            'processing_time_sec': 0.0,
            'glb_path': Path(glb_path).name,
            'ply_path': "model.ply",
            'depth_map_path': "depth_map.png",
        }
        
        with state_lock:
            pipeline_state["result"] = result_dict
            pipeline_state["status"] = "done"
            pipeline_state["progress"] = "Complete"
        
        logger.info("Pipeline completed successfully.")
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
