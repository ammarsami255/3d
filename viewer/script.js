/* ═══════════════════════════════════════════════════════════════════
   Antigravity — Three.js 3D Viewer & Upload Controller
   ═══════════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── DOM References ────────────────────────────────────────────
    const $  = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const uploadScreen     = $("#upload-screen");
    const processingScreen = $("#processing-screen");
    const viewerScreen     = $("#viewer-screen");
    const uploadZone       = $("#upload-zone");
    const fileInput        = $("#file-input");
    const previewImage     = $("#preview-image");
    const uploadPreview    = $("#upload-preview");
    const previewClear     = $("#preview-clear");
    const btnProcess       = $("#btn-process");
    const processingStatus = $("#processing-status");
    const navStatus        = $("#nav-status");
    const threeCanvas      = $("#three-canvas");
    const canvasWrapper    = $("#viewer-canvas-wrapper");

    // Info panel elements
    const infoSceneType  = $("#info-scene-type");
    const infoDepthRange = $("#info-depth-range");
    const infoPointCount = $("#info-point-count");
    const infoMeshFaces  = $("#info-mesh-faces");
    const infoProcTime   = $("#info-proc-time");
    const depthMapImg    = $("#depth-map-img");
    const sourceImg      = $("#source-img");
    const btnDownloadGlb = $("#btn-download-glb");
    const btnNewImage    = $("#btn-new-image");

    // Viewer control buttons
    const btnResetCamera     = $("#btn-reset-camera");
    const btnToggleWireframe = $("#btn-toggle-wireframe");
    const btnToggleAutorotate = $("#btn-toggle-autorotate");
    const btnFullscreen       = $("#btn-fullscreen");

    let selectedFile   = null;
    let uploadFilename = null;
    let pollInterval   = null;

    // ── Three.js State ────────────────────────────────────────────
    let scene, camera, renderer, controls, loadedModel;
    let wireframeMode = false;
    let autoRotate    = true;

    // ══════════════════════════════════════════════════════════════
    // SCREEN MANAGEMENT
    // ══════════════════════════════════════════════════════════════

    function showScreen(screen) {
        $$(".screen").forEach(s => s.classList.remove("active"));
        screen.classList.add("active");
    }

    function setNavStatus(text, state = "idle") {
        const dot  = navStatus.querySelector(".status-dot");
        const span = navStatus.querySelector(".status-text");
        dot.className = "status-dot " + state;
        span.textContent = text;
    }

    // ══════════════════════════════════════════════════════════════
    // FILE UPLOAD
    // ══════════════════════════════════════════════════════════════

    // Click to browse
    uploadZone.addEventListener("click", (e) => {
        if (e.target === previewClear || e.target.closest(".preview-clear")) return;
        if (!uploadZone.classList.contains("has-file")) {
            fileInput.click();
        }
    });

    // File selected via input
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear preview
    previewClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    function handleFile(file) {
        // Validate type
        const validTypes = ["image/jpeg", "image/png", "image/heic", "image/heif", "image/webp"];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|heic|heif|webp)$/i)) {
            showError("Please upload a JPG, PNG, or HEIC image.");
            return;
        }

        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadPreview.style.display = "block";
            uploadZone.classList.add("has-file");
            btnProcess.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        selectedFile = null;
        uploadFilename = null;
        previewImage.src = "";
        uploadPreview.style.display = "none";
        uploadZone.classList.remove("has-file");
        btnProcess.disabled = true;
        fileInput.value = "";
    }

    // ══════════════════════════════════════════════════════════════
    // PROCESS BUTTON → UPLOAD & START PIPELINE
    // ══════════════════════════════════════════════════════════════

    btnProcess.addEventListener("click", async () => {
        if (!selectedFile) return;

        btnProcess.disabled = true;
        showScreen(processingScreen);
        setNavStatus("Processing", "processing");
        resetPipelineSteps();

        try {
            const formData = new FormData();
            formData.append("image", selectedFile, selectedFile.name);

            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();

            if (data.error) {
                throw new Error(data.error);
            }

            uploadFilename = data.filename;
            startPolling();

        } catch (err) {
            showError("Upload failed: " + err.message);
            showScreen(uploadScreen);
            setNavStatus("Error", "error");
            btnProcess.disabled = false;
        }
    });

    // ══════════════════════════════════════════════════════════════
    // PIPELINE STATUS POLLING
    // ══════════════════════════════════════════════════════════════

    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        
        let elapsed = 0;
        pollInterval = setInterval(async () => {
            elapsed += 1;
            try {
                const res  = await fetch("/api/status");
                const data = await res.json();

                processingStatus.textContent = data.progress || "Working...";
                updatePipelineSteps(data.progress, elapsed);

                if (data.status === "done") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    onPipelineComplete(data.result);
                } else if (data.status === "error") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    showError("Pipeline error: " + (data.error || "Unknown error"));
                    showScreen(uploadScreen);
                    setNavStatus("Error", "error");
                    btnProcess.disabled = false;
                }
            } catch (e) {
                // Network hiccup, keep polling
            }
        }, 1500);
    }

    function resetPipelineSteps() {
        $$(".pipeline-step").forEach(s => {
            s.classList.remove("active", "done");
        });
        $("#step-analyze").classList.add("active");
    }

    function updatePipelineSteps(progress, elapsed) {
        const steps = [
            { id: "step-analyze",     keywords: ["analyz", "scene", "classify"], minTime: 0 },
            { id: "step-depth",       keywords: ["depth", "inference", "model", "loading"], minTime: 3 },
            { id: "step-pointcloud",  keywords: ["point cloud", "construct", "back-project"], minTime: 15 },
            { id: "step-mesh",        keywords: ["clean", "mesh", "reconstruct", "poisson", "outlier"], minTime: 25 },
            { id: "step-export",      keywords: ["export", "glb", "saving"], minTime: 35 },
        ];

        const progressLower = (progress || "").toLowerCase();
        
        // Find active step based on progress text or elapsed time
        let activeIdx = 0;
        for (let i = 0; i < steps.length; i++) {
            if (steps[i].keywords.some(kw => progressLower.includes(kw))) {
                activeIdx = i;
            } else if (elapsed > steps[i].minTime && i > activeIdx) {
                // Time-based fallback estimate
                activeIdx = i;
            }
        }

        steps.forEach((step, i) => {
            const el = document.getElementById(step.id);
            el.classList.remove("active", "done");
            if (i < activeIdx) {
                el.classList.add("done");
            } else if (i === activeIdx) {
                el.classList.add("active");
            }
        });
    }

    // ══════════════════════════════════════════════════════════════
    // PIPELINE COMPLETE → LOAD 3D VIEWER
    // ══════════════════════════════════════════════════════════════

    function onPipelineComplete(result) {
        setNavStatus("Ready", "idle");

        // Populate info panel
        const sceneLabel = (result.scene_type || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
        infoSceneType.textContent  = sceneLabel;
        infoDepthRange.textContent = `${result.depth_range_meters[0]}m – ${result.depth_range_meters[1]}m`;
        infoPointCount.textContent = result.point_count.toLocaleString();
        infoMeshFaces.textContent  = result.mesh_faces.toLocaleString();
        infoProcTime.textContent   = `${result.processing_time_sec}s`;

        // Depth map preview
        depthMapImg.src = "/output/depth_map.png";
        depthMapImg.onerror = () => { $("#depth-preview").style.display = "none"; };

        // Source image
        if (uploadFilename) {
            sourceImg.src = `/uploads/${uploadFilename}`;
        }

        // Download link
        btnDownloadGlb.href     = "/output/model.glb";
        btnDownloadGlb.download = "antigravity_model.glb";

        // Mark all steps done
        $$(".pipeline-step").forEach(s => {
            s.classList.remove("active");
            s.classList.add("done");
        });

        // Brief delay for visual satisfaction, then switch
        setTimeout(() => {
            showScreen(viewerScreen);
            initThreeJS();
            loadGLB("/output/model.glb");
        }, 600);
    }

    // ══════════════════════════════════════════════════════════════
    // THREE.JS 3D VIEWER
    // ══════════════════════════════════════════════════════════════

    function initThreeJS() {
        if (renderer) {
            // Already initialized, just resize
            onResize();
            return;
        }

        // Scene
        scene = new THREE.Scene();

        // Dark gradient background
        const bgColor = new THREE.Color(0x111114);
        scene.background = bgColor;
        scene.fog = new THREE.FogExp2(0x111114, 0.035);

        // Camera
        const aspect = canvasWrapper.clientWidth / canvasWrapper.clientHeight;
        camera = new THREE.PerspectiveCamera(50, aspect, 0.01, 1000);
        camera.position.set(4, 3, 6);

        // Renderer
        renderer = new THREE.WebGLRenderer({
            canvas: threeCanvas,
            antialias: true,
            alpha: false,
        });
        renderer.setSize(canvasWrapper.clientWidth, canvasWrapper.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.autoRotate = autoRotate;
        controls.autoRotateSpeed = 1.0;
        controls.minDistance = 1;
        controls.maxDistance = 30;
        controls.target.set(0, 0, 0);

        // Lighting — premium 3-point + ambient setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        // Key light (warm, from upper-right)
        const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.2);
        keyLight.position.set(5, 8, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 30;
        keyLight.shadow.camera.left = -10;
        keyLight.shadow.camera.right = 10;
        keyLight.shadow.camera.top = 10;
        keyLight.shadow.camera.bottom = -10;
        scene.add(keyLight);

        // Fill light (cool, from left)
        const fillLight = new THREE.DirectionalLight(0xe6eeff, 0.5);
        fillLight.position.set(-4, 3, -2);
        scene.add(fillLight);

        // Rim light (from behind)
        const rimLight = new THREE.DirectionalLight(0xc4b5fd, 0.4);
        rimLight.position.set(0, 2, -6);
        scene.add(rimLight);

        // Hemisphere for ambient fill
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
        scene.add(hemiLight);

        // Ground grid (subtle)
        const gridHelper = new THREE.GridHelper(20, 40, 0x333344, 0x222233);
        gridHelper.material.opacity = 0.3;
        gridHelper.material.transparent = true;
        gridHelper.position.y = -2.5;
        scene.add(gridHelper);

        // Resize handler
        window.addEventListener("resize", onResize);

        // Animation loop
        animate();

        // Fade out hint after 5s
        setTimeout(() => {
            const hint = $("#viewer-hint");
            if (hint) hint.style.opacity = "0";
        }, 5000);
    }

    function onResize() {
        if (!camera || !renderer) return;
        const w = canvasWrapper.clientWidth;
        const h = canvasWrapper.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }

    function animate() {
        requestAnimationFrame(animate);
        if (controls) controls.update();
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
    }

    function loadGLB(url) {
        const loader = new THREE.GLTFLoader();

        loader.load(
            url,
            (gltf) => {
                // Remove previous model
                if (loadedModel) {
                    scene.remove(loadedModel);
                    loadedModel.traverse((child) => {
                        if (child.geometry) child.geometry.dispose();
                        if (child.material) {
                            if (Array.isArray(child.material)) {
                                child.material.forEach(m => m.dispose());
                            } else {
                                child.material.dispose();
                            }
                        }
                    });
                }

                loadedModel = gltf.scene;

                // Enhance materials for premium look
                loadedModel.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        // If vertex colors exist, enhance the material
                        if (child.geometry.attributes.color) {
                            child.material = new THREE.MeshStandardMaterial({
                                vertexColors: true,
                                metalness: 0.05,
                                roughness: 0.7,
                                side: THREE.DoubleSide,
                            });
                        }
                    }
                });

                scene.add(loadedModel);

                // Auto-frame the model
                frameModel(loadedModel);
            },
            (progress) => {
                // Loading progress
                if (progress.total > 0) {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    processingStatus.textContent = `Loading 3D model... ${pct}%`;
                }
            },
            (error) => {
                console.error("GLB load error:", error);
                showError("Failed to load 3D model.");
            }
        );
    }

    function frameModel(model) {
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const distance = maxDim * 2;

        controls.target.copy(center);
        camera.position.set(
            center.x + distance * 0.7,
            center.y + distance * 0.5,
            center.z + distance * 0.7
        );
        camera.lookAt(center);
        controls.update();
    }

    // ══════════════════════════════════════════════════════════════
    // VIEWER CONTROLS
    // ══════════════════════════════════════════════════════════════

    btnResetCamera.addEventListener("click", () => {
        if (loadedModel) frameModel(loadedModel);
    });

    btnToggleWireframe.addEventListener("click", () => {
        wireframeMode = !wireframeMode;
        btnToggleWireframe.classList.toggle("active", wireframeMode);
        if (loadedModel) {
            loadedModel.traverse((child) => {
                if (child.isMesh && child.material) {
                    child.material.wireframe = wireframeMode;
                }
            });
        }
    });

    btnToggleAutorotate.addEventListener("click", () => {
        autoRotate = !autoRotate;
        btnToggleAutorotate.classList.toggle("active", autoRotate);
        if (controls) controls.autoRotate = autoRotate;
    });

    // Set initial active state
    btnToggleAutorotate.classList.add("active");

    btnFullscreen.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            canvasWrapper.requestFullscreen().catch(() => {});
        } else {
            document.exitFullscreen();
        }
    });

    // ── New Image button ──────────────────────────────────────────
    btnNewImage.addEventListener("click", () => {
        clearFile();
        showScreen(uploadScreen);
        setNavStatus("Ready", "idle");
    });

    // ══════════════════════════════════════════════════════════════
    // ERROR TOAST
    // ══════════════════════════════════════════════════════════════

    function showError(message) {
        let toast = $(".error-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.className = "error-toast";
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        
        // Trigger show
        requestAnimationFrame(() => {
            toast.classList.add("visible");
            setTimeout(() => {
                toast.classList.remove("visible");
            }, 5000);
        });
    }

    // ══════════════════════════════════════════════════════════════
    // INIT
    // ══════════════════════════════════════════════════════════════

    // On page load, check if there's already a completed result
    (async function checkExistingResult() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            if (data.status === "done" && data.result) {
                onPipelineComplete(data.result);
            }
        } catch (e) {
            // Server not running or no existing result — stay on upload screen
        }
    })();

})();
