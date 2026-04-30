# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:22:22 2025

@author: Aaron
"""
import cv2
import mediapipe as mp
import numpy as np
from isometric_camera_setup import IsometricCameraSetup
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            image_size=(100, 100),
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
                plt.close()
            else:
                plt.show()
                plt.close()

    def load_animation_sequence(self, numpy_array):
        """Loads animation frames from NumPy array"""
        self.frames = numpy_array
        return True

    def detect_poses_sequence(self):
        """Detects poses in all frames for all perspectives"""
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
                    frame_poses.append(landmarks_pixel)
                else:
                    print(f"No pose detected in perspective {perspective_idx+1}, frame {frame_idx+1}")
                    frame_poses.append(np.zeros((33, 3)))

            self.poses_sequence.append(np.array(frame_poses))

        self.poses_sequence = np.array(self.poses_sequence)
        print("Animation sequence pose detection completed")

    def triangulate_sequence(self):
        """Triangulate 3D poses for the entire sequence"""
        if self.poses_sequence is None:
            raise ValueError("No poses detected yet. Run detect_poses_sequence first.")

        triangulated_sequence = []

        for frame_idx, frame_poses in enumerate(self.poses_sequence):
            print(f"Triangulating frame {frame_idx+1}/{len(self.poses_sequence)}")

            frame_3d = []
            for landmark_idx in range(33):
                point_2d_views = {}

                for view_idx, pose in enumerate(frame_poses):
                    if not np.all(pose[landmark_idx] == 0):
                        point_2d_views[f'ISO-{view_idx+1}'] = pose[landmark_idx][:2]

                if len(point_2d_views) >= 2:
                    point_3d = self.camera_setup.triangulate_point(point_2d_views)
                    frame_3d.append(point_3d)
                else:
                    frame_3d.append(np.zeros(3))

            triangulated_sequence.append(np.array(frame_3d))

        return np.array(triangulated_sequence)

def main():
    try:
        #     try:
        import os  
        os.makedirs('pose_frames', exist_ok=True)
        print("Loading animation sequence...")
        images = np.load('critters_upscaled_edsr.npy')
        print(f"Loaded animation sequence shape: {images.shape}")

        triangulator = PoseTriangulator()
        triangulator.load_animation_sequence(images)
        triangulator.detect_poses_sequence()

        triangulated_poses = triangulator.triangulate_sequence()

        # Save the triangulated sequence
        np.save('triangulated_sequence.npy', triangulated_poses)
        print(f"Saved triangulated sequence with shape: {triangulated_poses.shape}")

        # Visualize the sequence
        triangulator.visualize_sequence(triangulated_poses, save_path='pose_frames')

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()