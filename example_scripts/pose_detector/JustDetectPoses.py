# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:28:24 2025

@author: Aaron
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class PoseProcessor:
    def __init__(self):
        print("Initializing MediaPipe Pose Detector...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # More stable for single images
            model_complexity=1,      # Lower complexity
            min_detection_confidence=0.5
        )
        print("MediaPipe initialized successfully")

    def process_frame(self, frame_idx=0):
        """Process a single frame from all perspectives"""
        try:
            # Step 1: Load data
            print("\nStep 1: Loading image data...")
            frames = np.load('critters_upscaled_edsr.npy')
            print(f"Data loaded successfully. Shape: {frames.shape}")

            # Step 2: Create visualization figure
            print("\nStep 2: Setting up visualization...")
            plt.figure(figsize=(15, 10))

            # Step 3: Process each perspective
            print("\nStep 3: Processing each perspective...")
            for perspective_idx in range(6):
                print(f"\nProcessing perspective {perspective_idx + 1}/6")

                # Get image for this perspective
                image = frames[perspective_idx][frame_idx]
                h, w = image.shape[:2]
                print(f"Image shape: {image.shape}")

                # Convert to RGB for MediaPipe
                print("Converting image to RGB...")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect pose
                print("Detecting pose...")
                results = self.pose.process(image_rgb)

                # Draw pose on image
                print("Drawing pose overlay...")
                annotated_image = image.copy()
                if results.pose_landmarks:
                    print("Pose detected successfully")
                    # Convert landmarks to pixel coordinates
                    landmarks_px = []
                    for landmark in results.pose_landmarks.landmark:
                        px = int(landmark.x * w)
                        py = int(landmark.y * h)
                        landmarks_px.append([px, py])
                    landmarks_px = np.array(landmarks_px)

                    # Draw connections
                    connections = [
                        # Torso
                        (11, 12), (11, 23), (12, 24), (23, 24),
                        # Arms
                        (11, 13), (13, 15), (12, 14), (14, 16),
                        # Legs
                        (23, 25), (25, 27), (24, 26), (26, 28)
                    ]

                    for start_idx, end_idx in connections:
                        if (landmarks_px[start_idx] > 0).all() and (landmarks_px[end_idx] > 0).all():
                            cv2.line(annotated_image,
                                   tuple(landmarks_px[start_idx]),
                                   tuple(landmarks_px[end_idx]),
                                   (0, 255, 0), 2)

                    # Draw landmarks
                    for point in landmarks_px:
                        if (point > 0).all():
                            cv2.circle(annotated_image, tuple(point), 3, (255, 0, 0), -1)
                else:
                    print("No pose detected for this perspective")

                # Add to subplot
                print("Adding to visualization grid...")
                plt.subplot(2, 3, perspective_idx + 1)
                plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                plt.title(f'View {perspective_idx + 1}')
                plt.axis('off')

            print("\nStep 4: Finalizing visualization...")
            plt.tight_layout()

            print("\nStep 5: Showing results...")
            plt.show()

            print("\nProcessing completed successfully")

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            raise

def main():
    print("Starting main processing...")
    processor = PoseProcessor()
    print("\nProcessing frame 0...")
    processor.process_frame(1)
    print("\nMain processing completed")

if __name__ == "__main__":
    main()