# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:33:55 2025

@author: Aaron
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class Animation2DPoseDetector:
    def __init__(self):
        """Initializes the Pose Detector with MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frames = []

    def normalized_to_pixel_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        """Converts normalized coordinates to pixel coordinates"""
        pixel_x = int(normalized_x * image_width)
        pixel_y = int(normalized_y * image_height)
        # Ensure coordinates are within image bounds
        pixel_x = max(0, min(pixel_x, image_width - 1))
        pixel_y = max(0, min(pixel_y, image_height - 1))
        return pixel_x, pixel_y

    def draw_pose_landmarks(self, image, landmarks):
        """Draw pose landmarks using pixel coordinates"""
        h, w = image.shape[:2]

        # Convert landmarks to pixel coordinates
        landmarks_pixel = []
        for landmark in landmarks.landmark:
            pixel_x, pixel_y = self.normalized_to_pixel_coordinates(landmark.x, landmark.y, w, h)
            landmarks_pixel.append((pixel_x, pixel_y))

        # Draw the connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            if (0 <= start_idx < len(landmarks_pixel) and
                0 <= end_idx < len(landmarks_pixel)):
                cv2.line(image,
                        landmarks_pixel[start_idx],
                        landmarks_pixel[end_idx],
                        (245, 66, 230),  # Pink color
                        thickness=2)

        # Draw the landmarks
        for landmark in landmarks_pixel:
            cv2.circle(image,
                      landmark,
                      radius=2,
                      color=(245, 117, 66),  # Orange color
                      thickness=-1)  # Filled circle

        return image

    def load_animation_sequence(self, numpy_array):
        """Loads animation frames from NumPy array"""
        self.frames = numpy_array
        return True

    def process_animation_sequence(self):
        """Process and visualize the entire animation sequence"""
        n_frames = self.frames.shape[1]
        n_perspectives = self.frames.shape[0]

        for frame_idx in range(n_frames):
            print(f"Processing frame {frame_idx+1}/{n_frames}")

            # Create a figure for this frame's perspectives
            fig = plt.figure(figsize=(15, 5))
            fig.suptitle(f'Frame {frame_idx+1} - All Perspectives')

            # Process each perspective
            for perspective_idx in range(n_perspectives):
                image = self.frames[perspective_idx][frame_idx].copy()

                # Process the image
                results = self.pose.process(image)

                # Draw the pose detection using pixel coordinates
                if results.pose_landmarks:
                    image = self.draw_pose_landmarks(image, results.pose_landmarks)

                # Add to plot
                plt.subplot(1, n_perspectives, perspective_idx + 1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f'Perspective {perspective_idx + 1}')
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            # Optional: add a small delay to make visualization more manageable
            plt.pause(0.5)

def main():
    try:
        # Load animation sequence
        print("Loading animation sequence...")
        images = np.load('critters_upscaled_edsr.npy')
        print(f"Loaded animation sequence shape: {images.shape}")

        # Create and use detector
        detector = Animation2DPoseDetector()
        detector.load_animation_sequence(images)

        # Process and visualize the complete animation sequence
        detector.process_animation_sequence()

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()