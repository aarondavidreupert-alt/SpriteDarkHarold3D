# SpriteDarkHarold3D

Fallout3D Pipeline Tool — Summary
19 files, ~4400 lines — launch with python fallout3d-pipeline/main.py

Pipeline backend (pipeline/)
File	What it does
isometric_camera_setup5.py	Cleaned-up IsometricCameraSetup — 6 cameras at 60° intervals, projection/triangulation/back-projection
pose_triangulator.py	PoseTriangulator — MediaPipe detection across all 6 views, left/right flip correction, weighted DLT triangulation, QThread-friendly progress callbacks
pose_library.py	PoseLibrary — cosine-similarity pose matching, confidence-weighted Mahalanobis-outlier-excluding averaging, JSON save/load
mesh_fitter.py	MeshFitter — OBJ template loading, inverse-distance LBS skinning weights, per-frame mesh deformation
normal_map_baker.py	NormalMapBaker — Sobel-gradient tangent-space normals + depth maps, batch (6 × N) baking
gltf_exporter.py	GLTFExporter — valid .glb with mesh, armature (33 joints), animation tracks, UV layout via pygltflib
GUI (gui/)
Tab	Key features
1 Asset Loader	.npy/.png/.frm loading (FRM reader built-in), category dropdown, thumbnail grid 6 dirs × N frames, multi-character list
2 Pose Editor	6-view display, draggable QGraphicsEllipseItem landmarks, confidence heatmap overlay, arrow-key frame nav, "apply to all frames/chars"
3 3D Reconstruction	pyqtgraph OpenGL skeleton viewer, reprojection-error colour coding (red→green per landmark), error table
4 Pose Library	Multi-character 3D overlay (different colours), master skeleton computation, pose-match table, library JSON I/O
5 Mesh & Normals	Template fitting, per-bone skinning weight heatmap, normal map preview per direction/frame
6 Export	GLB + animation_data.json + normal map PNGs + pose library JSON, live preview, link to gltf-viewer
Install
pip install PyQt6 pyqtgraph PyOpenGL mediapipe numpy opencv-python pygltflib
