# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:20:56 2025

@author: Aaron
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:07:25 2025

@author: Aaron
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:37:53 2025

@author: Aaron
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:22:22 2025

@author: Aaron
"""
import cv2
import mediapipe as mp
import numpy as np
from isometric_camera_setup5 import IsometricCameraSetup
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define body part connections and colors
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),  # shoulders to hips
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    # Legs
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)  # face outline
]

COLORS = {
    'head': (255, 0, 0),    # Red
    'torso': (0, 255, 0),   # Green
    'arms': (0, 0, 255),    # Blue
    'legs': (255, 255, 0),  # Yellow
    'face': (255, 0, 255)   # Magenta
}

LANDMARK_COLORS = {i: 'face' for i in range(0, 11)}
LANDMARK_COLORS.update({i: 'torso' for i in [11, 12, 23, 24]})
LANDMARK_COLORS.update({i: 'arms' for i in [13, 14, 15, 16]})
LANDMARK_COLORS.update({i: 'legs' for i in [25, 26, 27, 28]})

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized coordinates to pixel coordinates"""
    pixel_x = min(math.floor(normalized_x * image_width), image_width - 1)
    pixel_y = min(math.floor(normalized_y * image_height), image_height - 1)
    return pixel_x, pixel_y

class PoseTriangulator:
    def __init__(self):
        """Initialize MediaPipe and IsometricCameraSetup"""
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize isometric camera setup
        self.camera_setup = IsometricCameraSetup(
            radius=2.0,
            height=1.5,
            focal_length=500,
            image_size=(400, 400),
            subject_height=0.3
        )

        self.frames = None
        self.poses_sequence = None

    def visualize_3d_pose(self, points, ax=None, frame_idx=None):
        """Visualize 3D pose with colored landmarks and skeleton connections"""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
    
        # Plot landmarks
        for i, point in enumerate(points):
            color = COLORS[LANDMARK_COLORS.get(i, 'torso')]
            ax.scatter(point[0], point[1], point[2], c=[np.array(color)/255], s=50)
    
        # Plot connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            color = COLORS[LANDMARK_COLORS.get(start_idx, 'torso')]
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    c=np.array(color)/255)
    
        # Set consistent view limits based on camera setup
        # limit = 1.2 * max(self.camera_setup.radius, self.camera_setup.height)
        limit = 1.2 
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
    
        # Set consistent view angle
        ax.view_init(elev=20, azim=45)
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        if frame_idx is not None:
            ax.set_title(f'Frame {frame_idx}')
    
        return ax
    
    def visualize_sequence(self, triangulated_points, save_path=None):
        """Visualize or save the entire sequence of 3D poses"""
        for frame_idx, frame_points in enumerate(triangulated_points):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
    
            self.visualize_3d_pose(frame_points, ax, frame_idx)
    
            if save_path:
                plt.savefig(f'{save_path}/frame_{frame_idx:03d}.png')
                plt.show()
                plt.close()
            else:
                plt.show()
                plt.close()

    def load_animation_sequence(self, numpy_array):
        """Loads animation frames from NumPy array"""
        self.frames = numpy_array
        return True
    
    def check_and_correct_pose(self, pose_2d, perspective_idx):
        """
        Check and correct pose orientation based on camera perspective
        """
        # Landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
    
        # Camera angle for this perspective (0, 60, 120, 180, 240, 300 degrees)
        camera_angle = perspective_idx * 60  # degrees
        expected_direction = np.array([np.cos(np.radians(camera_angle)),
                                     np.sin(np.radians(camera_angle))])
    
        corrected_pose = pose_2d.copy()
    
        def check_segment_flip(left_point, right_point):
            """Check if a segment needs flipping"""
            if np.all(left_point == 0) or np.all(right_point == 0):
                return False
    
            segment_vector = right_point[:2] - left_point[:2]
            # Changed: Use dot product instead of cross product
            dot_product = np.dot(segment_vector, expected_direction)
            return dot_product < 0  # Flip if pointing wrong way
    
        # Check shoulders
        shoulder_flip_needed = check_segment_flip(
            pose_2d[LEFT_SHOULDER],
            pose_2d[RIGHT_SHOULDER]
        )
    
        # Points to swap if flip is needed
        points_to_flip = [
            (11, 12),  # shoulders
            (13, 14),  # elbows
            (15, 16),  # wrists
            (23, 24),  # hips
            # Face points
            (7, 8),    # ears
            (9, 10),   # mouth corners
            (1, 4),    # eye outer corners
            (2, 5),    # eye inner corners
            (3, 6),    # eye centers
        ]
    
        # Apply flip if needed
        if shoulder_flip_needed:
            for left, right in points_to_flip:
                corrected_pose[left], corrected_pose[right] = \
                    corrected_pose[right].copy(), corrected_pose[left].copy()
    
        return corrected_pose
    # def check_and_correct_pose(self, pose_2d, perspective_idx):
    #     """
    #     Check and correct shoulder and hip flips independently
    
    #     Args:
    #         pose_2d: 2D pose landmarks for a single view
    #         perspective_idx: Index of the camera perspective (0-5)
    
    #     Returns:
    #         Corrected pose
    #     """
    #     # Landmark indices
    #     LEFT_SHOULDER = 11
    #     RIGHT_SHOULDER = 12
    #     LEFT_HIP = 23
    #     RIGHT_HIP = 24
    
    #     # Camera angle for this perspective
    #     camera_angle = perspective_idx * 60  # degrees
    #     expected_direction = np.array([np.cos(np.radians(camera_angle)),
    #                                  np.sin(np.radians(camera_angle))])
    
    #     corrected_pose = pose_2d.copy()
    
    #     def check_segment_flip(left_point, right_point):
    #         """Check if a segment (shoulders or hips) needs flipping"""
    #         if np.all(left_point == 0) or np.all(right_point == 0):
    #             return False
    
    #         segment_vector = right_point[:2] - left_point[:2]
    #         cross_product = np.cross(segment_vector, expected_direction)
    #         return cross_product < 0  # True if needs flipping
    
    #     # Check shoulders
    #     shoulder_flip_needed = check_segment_flip(
    #         pose_2d[LEFT_SHOULDER],
    #         pose_2d[RIGHT_SHOULDER]
    #     )
    
    #     # Check hips
    #     hip_flip_needed = check_segment_flip(
    #         pose_2d[LEFT_HIP],
    #         pose_2d[RIGHT_HIP]
    #     )
    
    #     # Define points to swap for each segment
    #     shoulder_points = [
    #         (11, 12),  # shoulders
    #         (13, 14),  # elbows
    #         (15, 16),  # wrists
    #         # Face points
    #         (7, 8),    # ears
    #         (9, 10),   # mouth corners
    #         (1, 4),    # eye outer corners
    #         (2, 5),    # eye inner corners
    #         (3, 6),    # eye centers
    #     ]
    
    #     hip_points = [
    #         (23, 24),  # hips
    #         (25, 26),  # knees
    #         (27, 28),  # ankles
    #     ]
    
    #     # Apply corrections independently
    #     if shoulder_flip_needed:
    #         for left, right in shoulder_points:
    #             corrected_pose[left], corrected_pose[right] = \
    #                 corrected_pose[right].copy(), corrected_pose[left].copy()
    
    #     if hip_flip_needed:
    #         for left, right in hip_points:
    #             corrected_pose[left], corrected_pose[right] = \
    #                 corrected_pose[right].copy(), corrected_pose[left].copy()
    
    #     return corrected_pose
    
    def detect_poses_sequence(self):
        """Detects poses in all frames for all perspectives with independent flip checking"""
        print("Starting pose detection for animation sequence...")
        n_frames = self.frames.shape[1]
        n_perspectives = self.frames.shape[0]
    
        self.poses_sequence = []
    
        for frame_idx in range(n_frames):
            print(f"Processing frame {frame_idx+1}/{n_frames}")
            frame_poses = []
    
            for perspective_idx in range(n_perspectives):
                image = self.frames[perspective_idx][frame_idx]
                h, w = image.shape[:2]
    
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)
    
                if results.pose_landmarks:
                    landmarks_pixel = np.array([
                        [*normalized_to_pixel_coordinates(lm.x, lm.y, w, h), lm.z]
                        for lm in results.pose_landmarks.landmark
                    ])
    
                    # Check and correct flips independently
                    corrected_landmarks = self.check_and_correct_pose(
                        landmarks_pixel,
                        perspective_idx
                    )
                    frame_poses.append(corrected_landmarks)
                else:
                    print(f"No pose detected in perspective {perspective_idx+1}, frame {frame_idx+1}")
                    frame_poses.append(np.zeros((33, 3)))
    
            self.poses_sequence.append(np.array(frame_poses))
    
        self.poses_sequence = np.array(self.poses_sequence)
        print("Animation sequence pose detection completed")

    def triangulate_sequence(self):
        """Triangulate 3D poses for the entire sequence using landmark confidence weights"""
        if self.poses_sequence is None:
            raise ValueError("No poses detected yet. Run detect_poses_sequence first.")
    
        triangulated_sequence = []
    
        for frame_idx, frame_poses in enumerate(self.poses_sequence):
            print(f"Triangulating frame {frame_idx+1}/{len(self.poses_sequence)}")
    
            frame_3d = []
            for landmark_idx in range(33):
                point_2d_views = {}
                weights = {}
    
                for view_idx, pose in enumerate(frame_poses):
                    if not np.all(pose[landmark_idx] == 0):
                        point_2d_views[f'ISO-{view_idx+1}'] = pose[landmark_idx][:2]
                        # Use z value as confidence weight (MediaPipe provides this)
                        weights[f'ISO-{view_idx+1}'] = max(0.1, pose[landmark_idx][2])
    
                if len(point_2d_views) >= 2:
                    point_3d = self.camera_setup.triangulate_point(point_2d_views, weights)
                    frame_3d.append(point_3d)
                else:
                    frame_3d.append(np.zeros(3))
    
            triangulated_sequence.append(np.array(frame_3d))
    
        return np.array(triangulated_sequence)
    
    def overlay_pose_on_image(self, image, pose_2d, frame_idx, perspective_idx):
        """
        Overlay 2D pose on the original image
    
        Args:
            image: Original image
            pose_2d: 2D pose points to overlay
            frame_idx: Frame index for labeling
            perspective_idx: Perspective index (0-5)
        """
        # Create a copy of the image
        img_overlay = image.copy()
        if len(img_overlay.shape) == 2:  # If grayscale, convert to RGB
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_GRAY2RGB)
    
        # Define colors for different body parts
        colors = {
            'left_arm': (255, 0, 0),    # Blue
            'right_arm': (0, 0, 255),   # Red
            'left_leg': (0, 255, 0),    # Green
            'right_leg': (255, 255, 0), # Cyan
            'torso': (255, 0, 255)      # Magenta
        }
    
        # Define connections with their corresponding colors
        connections = [
            # Torso
            (11, 12, 'torso'), (12, 24, 'torso'), (24, 23, 'torso'), (23, 11, 'torso'),
            # Arms
            (11, 13, 'left_arm'), (13, 15, 'left_arm'),
            (12, 14, 'right_arm'), (14, 16, 'right_arm'),
            # Legs
            (23, 25, 'left_leg'), (25, 27, 'left_leg'),
            (24, 26, 'right_leg'), (26, 28, 'right_leg')
        ]
    
        # Draw connections
        for start_idx, end_idx, body_part in connections:
            if not (np.all(pose_2d[start_idx] == 0) or np.all(pose_2d[end_idx] == 0)):
                start_point = tuple(map(int, pose_2d[start_idx][:2]))
                end_point = tuple(map(int, pose_2d[end_idx][:2]))
                cv2.line(img_overlay, start_point, end_point, colors[body_part], 2)
    
        # Draw landmarks
        for i, point in enumerate(pose_2d):
            if not np.all(point == 0):
                point = tuple(map(int, point[:2]))
                cv2.circle(img_overlay, point, 3, (255, 255, 255), -1)
    
        # Add text labels
        cv2.putText(img_overlay, f'Frame: {frame_idx}, View: {perspective_idx+1}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
        return img_overlay
    
    def create_pose_visualization_sequence(self):
        """Create visualization sequence with back-projected poses overlaid on original images"""
        if self.frames is None or self.poses_sequence is None:
            raise ValueError("No frames or poses loaded")
    
        import os
        os.makedirs('overlay_frames', exist_ok=True)
    
        n_frames = len(self.poses_sequence)
        n_perspectives = len(self.poses_sequence[0])
    
        print(f"Creating overlay visualizations for {n_frames} frames...")
    
        # Triangulate all frames first
        triangulated_sequence = self.triangulate_sequence()
    
        for frame_idx in range(n_frames):
            # Get back-projected points for this frame
            back_projected_points = self.camera_setup.back_project_points(triangulated_sequence[frame_idx])
    
            for perspective_idx in range(n_perspectives):
                # Get original image
                original_image = self.frames[perspective_idx][frame_idx]
    
                # Create overlay using back-projected points
                overlay = self.overlay_pose_on_image(
                    original_image,
                    back_projected_points[perspective_idx],  # Now in same format as input landmarks
                    frame_idx,
                    perspective_idx
                )
    
                # Save overlay
                cv2.imwrite(
                    f'overlay_frames/frame_{frame_idx:03d}_view_{perspective_idx+1}.png',
                    overlay
                )
    
        print("Overlay visualization sequence created in 'overlay_frames' directory")   
        
    def create_comparison_view(self, frame_idx, triangulated_poses):
        """
        Create a side-by-side view of all perspectives with 3D pose for a given frame
        using back-projected points
        """
        import matplotlib.pyplot as plt
    
        # Create figure with subplots in a 2x4 grid
        fig = plt.figure(figsize=(20, 10))
    
        # 3D view in first position
        ax_3d = fig.add_subplot(241, projection='3d')
        self.visualize_3d_pose(triangulated_poses[frame_idx], ax_3d, frame_idx)
        ax_3d.set_title(f'3D Pose - Frame {frame_idx}')
    
        # Get back-projected points for this frame
        back_projected_points = self.camera_setup.back_project_points(triangulated_poses[frame_idx])
    
        # 2D views with overlays
        for i in range(6):
            ax = fig.add_subplot(2, 4, i + 3)
            original_image = self.frames[i][frame_idx]
            overlay = self.overlay_pose_on_image(
                original_image,
                back_projected_points[i],  # Using back-projected points instead of detected poses
                frame_idx,
                i
            )
            # Convert BGR to RGB for matplotlib
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            ax.imshow(overlay_rgb)
            ax.set_title(f'View {i+1} (Back-projected)')
            ax.axis('off')
    
        plt.tight_layout()
        plt.show()
        return fig
    
    def create_sequence_comparison(self, triangulated_poses, output_dir='comparison_frames'):
        """
        Create comparison views for the entire sequence
    
        Args:
            triangulated_poses: Array of triangulated 3D poses
            output_dir: Directory to save the comparison frames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
    
        n_frames = len(self.poses_sequence)
        print(f"Creating comparison views for {n_frames} frames...")
    
        for frame_idx in range(n_frames):
            print(f"Creating comparison view for frame {frame_idx+1}/{n_frames}")
            fig = self.create_comparison_view(frame_idx, triangulated_poses)
            plt.savefig(f'{output_dir}/comparison_frame_{frame_idx:03d}.png',
                       bbox_inches='tight', dpi=150)
            plt.close(fig)
    
        print("Comparison views created successfully!")
        
    def export_animation_data(self, triangulated_poses, filename='animation_data.json'):
        """
        Export pose data in JSON format
    
        Args:
            triangulated_poses: Array of triangulated 3D poses
            filename: Output JSON filename
        """
        import json
        data = {
            'metadata': {
                'total_frames': len(self.poses_sequence),
                'perspectives': len(self.poses_sequence[0]),
                'image_size': self.camera_setup.image_size,
                'landmarks_per_pose': 33
            },
            'poses_2d': self.poses_sequence.tolist(),
            'poses_3d': triangulated_poses.tolist()
        }
    
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
        print(f"Animation data exported to {filename}")        

    def visualize_comparison(self, frame_idx: int, perspective_idx: int, save_path: str = None):
        """
        Create a visualization comparing original and back-projected landmarks
    
        Args:
            frame_idx: Frame index
            perspective_idx: Perspective/camera index
            save_path: Optional path to save visualization
        """
        # Get original image and detected pose
        original_image = self.frames[perspective_idx][frame_idx].copy()
        original_pose = self.poses_sequence[frame_idx][perspective_idx]
    
        # Get triangulated 3D points for this frame
        triangulated_points = self.triangulate_sequence()[frame_idx]
    
        # Back-project the triangulated points
        back_projected = self.camera_setup.back_project_points(triangulated_points)
        back_projected_pose = back_projected[perspective_idx]
    
        # Create visualization
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
        # Draw original landmarks in blue
        for i, point in enumerate(original_pose):
            if not np.all(point == 0):
                pt = tuple(map(int, point[:2]))
                cv2.circle(original_image, pt, 3, (255, 0, 0), -1)  # Blue
    
        # Draw back-projected landmarks in red
        for i, point in enumerate(back_projected_pose):
            if not np.all(point == 0):
                pt = tuple(map(int, point[:2]))
                cv2.circle(original_image, pt, 3, (0, 0, 255), -1)  # Red
    
        # Draw connections between corresponding points
        for i in range(len(original_pose)):
            if not (np.all(original_pose[i] == 0) or np.all(back_projected_pose[i] == 0)):
                pt1 = tuple(map(int, original_pose[i][:2]))
                pt2 = tuple(map(int, back_projected_pose[i][:2]))
                cv2.line(original_image, pt1, pt2, (0, 255, 0), 1)  # Green
    
                # Calculate and display error
                error = np.linalg.norm(original_pose[i][:2] - back_projected_pose[i][:2])
                mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
                cv2.putText(original_image,
                           f"{error:.1f}",
                           mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.3,
                           (255, 255, 255),
                           1)
    
        # Add legend
        cv2.putText(original_image, "Blue: Original", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(original_image, "Red: Back-projected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(original_image, f"Frame {frame_idx}, View {perspective_idx+1}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        if save_path:
            cv2.imwrite(save_path, original_image)
    
        return original_image
    
    # def create_comparison_views(self):
    #     """Create comparison views for all frames and perspectives"""
    #     print("Creating comparison views...")
    #     print(f"Creating comparison views for {len(self.poses_sequence)} frames...")
    
    #     os.makedirs('comparison_frames', exist_ok=True)
    
    #     for frame_idx in range(len(self.poses_sequence)):
    #         print(f"Creating comparison view for frame {frame_idx+1}/{len(self.poses_sequence)}")
    #         for perspective_idx in range(len(self.poses_sequence[0])):
    #             save_path = f'comparison_frames/comparison_frame_{frame_idx:03d}_view_{perspective_idx+1}.png'
    #             self.visualize_comparison(frame_idx, perspective_idx, save_path)
    
    #     print("Comparison views created successfully!")
    
    def create_comparison_views(self):
        """Create comparison views for all frames with multiple perspectives in a grid"""
        print("Creating comparison views...")
        print(f"Creating comparison views for {len(self.poses_sequence)} frames...")
    
        os.makedirs('comparison_frames', exist_ok=True)
    
        for frame_idx in range(len(self.poses_sequence)):
            print(f"Creating comparison view for frame {frame_idx+1}/{len(self.poses_sequence)}")
    
            # Create a figure with 2x3 grid for 6 perspectives
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Frame {frame_idx+1} - All Perspectives', fontsize=16)
    
            # Flatten axes for easier iteration
            axes_flat = axes.flatten()
    
            for perspective_idx in range(len(self.poses_sequence[0])):
                # Get original image and create comparison overlay
                original_image = self.frames[perspective_idx][frame_idx]
                comparison = self.visualize_comparison(frame_idx, perspective_idx)
    
                # Display in the corresponding subplot
                axes_flat[perspective_idx].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
                axes_flat[perspective_idx].set_title(f'Perspective {perspective_idx+1}')
                axes_flat[perspective_idx].axis('off')
    
            # Adjust layout and save
            plt.tight_layout()
            save_path = f'comparison_frames/frame_{frame_idx:03d}_all_views.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
        print("Comparison views created successfully!")
    
    def visualize_comparison(self, frame_idx: int, perspective_idx: int, save_path: str = None):
        """
        Create a visualization comparing original and back-projected landmarks
    
        Args:
            frame_idx: Frame index
            perspective_idx: Perspective/camera index
            save_path: Optional path to save visualization
        """
        # Get original image and detected pose
        original_image = self.frames[perspective_idx][frame_idx].copy()
        original_pose = self.poses_sequence[frame_idx][perspective_idx]
    
        # Get triangulated 3D points for this frame
        triangulated_points = self.triangulate_sequence()[frame_idx]
    
        # Back-project the triangulated points
        back_projected = self.camera_setup.back_project_points(triangulated_points)
        back_projected_pose = back_projected[perspective_idx]
    
        # Create visualization
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
        # Draw original landmarks in blue
        for i, point in enumerate(original_pose):
            if not np.all(point == 0):
                pt = tuple(map(int, point[:2]))
                cv2.circle(original_image, pt, 3, (255, 0, 0), -1)  # Blue
    
        # Draw back-projected landmarks in red
        for i, point in enumerate(back_projected_pose):
            if not np.all(point == 0):
                pt = tuple(map(int, point[:2]))
                cv2.circle(original_image, pt, 3, (0, 0, 255), -1)  # Red
    
        # Draw connections between corresponding points
        for i in range(len(original_pose)):
            if not (np.all(original_pose[i] == 0) or np.all(back_projected_pose[i] == 0)):
                pt1 = tuple(map(int, original_pose[i][:2]))
                pt2 = tuple(map(int, back_projected_pose[i][:2]))
                cv2.line(original_image, pt1, pt2, (0, 255, 0), 1)  # Green
    
                # Calculate and display error
                error = np.linalg.norm(original_pose[i][:2] - back_projected_pose[i][:2])
                mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
                cv2.putText(original_image,
                           f"{error:.1f}",
                           mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.3,
                           (255, 255, 255),
                           1)
    
        # Add legend
        cv2.putText(original_image, "Blue: Original", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(original_image, "Red: Back-projected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
        if save_path:
            cv2.imwrite(save_path, original_image)
    
        return original_image

def main():
    try:
        # Load animation sequence
        print("Loading animation sequence...")
        images = np.load('critters_upscaled_edsr.npy')
        print(f"Loaded animation sequence shape: {images.shape}")

        # Initialize triangulator
        triangulator = PoseTriangulator()

        # Load and process sequence
        triangulator.load_animation_sequence(images)
        triangulator.detect_poses_sequence()

        # Triangulate the sequence
        triangulated_poses = triangulator.triangulate_sequence()

        # Create comparison views
        print("Creating comparison views...")
        triangulator.create_sequence_comparison(triangulated_poses)

        # Export animation data
        print("Exporting animation data...")
        triangulator.export_animation_data(triangulated_poses)

        print("Processing complete!")
        # Create comparison visualizations  
        os.makedirs('comparison_frames3', exist_ok=True)  
        for frame_idx in range(len(triangulator.poses_sequence)):  
            for perspective_idx in range(6):  
                comparison_image = triangulator.visualize_comparison(  
                    frame_idx,  
                    perspective_idx,  
                    f'comparison_frames3/comparison_frame_{frame_idx:03d}_view_{perspective_idx+1}.png'  
                )  

        print("Comparison visualizations created in 'comparison_frames' directory")  
        
        
        # Create comparison views  
        triangulator.create_comparison_views()  

        # Save results  
        np.save('triangulated_sequence.npy', triangulated_poses)  
        print(f"Saved triangulated sequence with shape: {triangulated_poses.shape}")  

        print("Processing complete!")
        
        
    # except Exception as e:  
    #     print(f"Error during processing: {str(e)}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()