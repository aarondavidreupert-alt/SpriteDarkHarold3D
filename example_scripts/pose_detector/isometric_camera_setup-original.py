# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:35:10 2025

@author: Aaron
"""
# -*- coding: utf-8 -*-
"""
Enhanced Isometric Camera Setup
Author: Claude
Date: Jan 30, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Union
import json


class IsometricCameraSetup:  
    def __init__(self,  
                 radius: float = 2.0,  
                 height: float = 1.5,  
                 focal_length: float = 500,  
                 image_size: Tuple[int, int] = (1000, 1000),  
                 subject_height: float = 0.3):  
        """  
        Initialize the isometric camera setup with corrected parameters.  
        """  
        self.radius = radius  
        self.height = height  
        self.focal_length = focal_length  
        self.image_size = image_size  
        self.subject_height = subject_height  
        self.principal_point = (image_size[0]/2, image_size[1]/2)  

        # Initialize intrinsic matrix  
        self.K = np.array([  
            [self.focal_length, 0, self.principal_point[0]],  
            [0, self.focal_length, self.principal_point[1]],  
            [0, 0, 1]  
        ])  

        self._initialize_cameras()

    def _initialize_cameras(self):
        """Initialize all camera parameters and matrices"""
        # Intrinsic matrix
        self.K = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])
        
        self.camera_views = []
        self.projection_matrices = []
        
        for i in range(6):
            angle = i * 60
            camera_params = self._compute_camera_parameters(angle)
            self.camera_views.append(camera_params)
            self.projection_matrices.append(camera_params['projection'])



    def get_camera_parameters(self, camera_index: int) -> Dict:
        """
        Get parameters for a specific camera.
        
        Args:
            camera_index: Index of the camera (0-5)
        
        Returns:
            Dictionary of camera parameters
        """
        if 0 <= camera_index < len(self.camera_views):
            return self.camera_views[camera_index]
        raise ValueError(f"Invalid camera index: {camera_index}")



    def _is_point_in_image(self, point_2d: np.ndarray) -> bool:
        """
        Check if a 2D point lies within image bounds.
        
        Args:
            point_2d: 2D point coordinates [x, y]
        
        Returns:
            Boolean indicating if point is within image bounds
        """
        return (0 <= point_2d[0] <= self.image_size[0] and 
                0 <= point_2d[1] <= self.image_size[1])

    def save_configuration(self, filename: str):
        """
        Save camera configuration to JSON file.
        
        Args:
            filename: Path to save configuration
        """
        config = {
            'parameters': {
                'radius': self.radius,
                'height': self.height,
                'focal_length': self.focal_length,
                'image_size': self.image_size,
                'subject_height': self.subject_height
            },
            'cameras': [
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in cam.items()}
                for cam in self.camera_views
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_configuration(cls, filename: str) -> 'IsometricCameraSetup':
        """
        Load camera configuration from JSON file.
        
        Args:
            filename: Path to configuration file
        
        Returns:
            IsometricCameraSetup instance
        """
        with open(filename, 'r') as f:
            config = json.load(f)
        
        setup = cls(**config['parameters'])
        return setup

    def visualize_camera_setup(self, test_points: Optional[List[np.ndarray]] = None):
        """
        Visualize the isometric camera setup with optional test points.
        
        Args:
            test_points: List of 3D points to visualize
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw subject point
        ax.scatter([0], [0], [self.subject_height], color='red', s=100, label='Subject')
        
        # Draw ground circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)
        circle_z = np.zeros_like(theta)
        ax.plot(circle_x, circle_y, circle_z, '--', color='gray', alpha=0.3)
        
        # Draw cameras and view cones
        colors = plt.cm.rainbow(np.linspace(0, 1, 6))
        for cam, color in zip(self.camera_views, colors):
            self._draw_camera(ax, cam, color)
        
        # Draw test points
        if test_points is not None:
            for i, point in enumerate(test_points):
                self._draw_test_point(ax, point, i)
        
        self._setup_axes(ax)
        plt.show()

    def _draw_camera(self, ax, cam: Dict, color: np.ndarray):
        """Helper method to draw a single camera"""
        pos = cam['position']
        look_at = cam['look_at']
        
        # Camera position
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=100)
        
        # View direction
        arrow = look_at - pos
        ax.quiver(pos[0], pos[1], pos[2],
                 arrow[0], arrow[1], arrow[2],
                 color=color, alpha=0.7,
                 arrow_length_ratio=0.1)
        
        # View cone
        self._draw_view_cone(ax, pos, look_at, color)
        
        # Label
        ax.text(pos[0], pos[1], pos[2],
               f'{cam["name"]}\n{cam["angle"]}°',
               fontsize=8)

    def _draw_view_cone(self, ax, pos: np.ndarray, look_at: np.ndarray, color: np.ndarray):
        """Helper method to draw camera view cone"""
        arrow = look_at - pos
        cone_length = np.linalg.norm(arrow)
        cone_angle = np.radians(30)
        cone_radius = cone_length * np.tan(cone_angle)
        
        # Compute cone base circle
        circle_points = 16
        angles = np.linspace(0, 2*np.pi, circle_points)
        
        forward = arrow / np.linalg.norm(arrow)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        for angle in angles:
            end_point = (look_at +
                        cone_radius * np.cos(angle) * right +
                        cone_radius * np.sin(angle) * up)
            ax.plot([pos[0], end_point[0]],
                   [pos[1], end_point[1]],
                   [pos[2], end_point[2]],
                   '-', color=color, alpha=0.1)

    def _draw_test_point(self, ax, point: np.ndarray, index: int):
        """Helper method to draw a test point"""
        ax.scatter(point[0], point[1], point[2],
                  color='green', s=100,
                  label=f'Test Point {index+1}')
        ax.plot([point[0], point[0]],
               [point[1], point[1]],
               [0, point[2]],
               ':', color='green', alpha=0.5)

    def _setup_axes(self, ax):
        """Helper method to setup plot axes"""
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Isometric Camera Setup (60° Intervals)')
        
        ax.set_box_aspect([1,1,1])
        
        limit = 1.2 * max(self.radius, self.height)
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
        
        plt.legend()
        

    def project_point_to_all_views(self, point: np.ndarray) -> List[Dict]:  
        """  
        Project a 3D point to all camera views with corrected projection.  
        """  
        results = []  
        point_h = np.append(point, 1)  # Make homogeneous  

        for view in self.camera_views:  
            P = view['projection']  
            point_2d_h = P @ point_h  

            if abs(point_2d_h[2]) > 1e-10:  
                point_2d = point_2d_h[:2] / point_2d_h[2]  
                valid = self._is_point_in_image(point_2d)  
            else:  
                point_2d = np.array([float('nan'), float('nan')])  
                valid = False  

            results.append({  
                'camera': view['name'],  
                'projection': point_2d,  
                'valid': valid  
            })  
        return results  


    
    def triangulate_point(self, image_points: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Triangulate a 3D point from multiple 2D projections using a simpler method.
        """
        A = []
    
        for view in self.camera_views:
            camera_name = view['name']
            if camera_name in image_points:
                x, y = image_points[camera_name]
                P = view['projection']
    
                # Each point contributes two equations
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])
    
        A = np.array(A)
    
        # Solve using SVD
        U, S, Vh = np.linalg.svd(A)
    
        # Get the point from the last row of Vh
        X = Vh[-1, :]
    
        # Convert to non-homogeneous coordinates
        X = X / X[3]
        point_3d = X[:3]
    
        return point_3d
    
    def _compute_camera_parameters(self, angle: float) -> Dict:
        """
        Compute parameters for a single camera with simplified projection matrix.
        """
        # Camera position
        x = self.radius * np.cos(np.radians(angle))
        y = self.radius * np.sin(np.radians(angle))
        z = self.height
        position = np.array([x, y, z])
    
        # Look-at point
        look_at = np.array([0, 0, self.subject_height])
    
        # Camera coordinate system
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
    
        right = np.cross(np.array([0, 0, 1]), forward)
        right = right / np.linalg.norm(right)
    
        up = np.cross(forward, right)
    
        # Rotation matrix (world to camera)
        R = np.vstack((right, up, -forward))
    
        # Translation
        t = -R @ position
    
        # Projection matrix
        P = self.K @ np.hstack((R, t.reshape(3,1)))
    
        return {
            'name': f'ISO-{angle//60 + 1}',
            'angle': angle,
            'position': position,
            'look_at': look_at,
            'rotation': R,
            'translation': t,
            'projection': P
        }
    def verify_triangulation(self, original_point: np.ndarray, projections: List[Dict]) -> Dict:
        """
        Verify triangulation by comparing original and reconstructed points,
        including reprojection error testing.
        """
        # Create dictionary of valid projections
        valid_projections = {
            proj['camera']: proj['projection']
            for proj in projections
            if proj['valid']
        }
    
        # Triangulate point
        reconstructed_point = self.triangulate_point(valid_projections)
    
        # Calculate 3D error
        error_3d = np.linalg.norm(original_point - reconstructed_point)
    
        # Calculate reprojection errors
        reprojection_errors = {}
        reprojs = self.project_point_to_all_views(reconstructed_point)
    
        for orig_proj, reproj in zip(projections, reprojs):
            if orig_proj['valid'] and reproj['valid']:
                camera_name = orig_proj['camera']
                original_2d = orig_proj['projection']
                reprojected_2d = reproj['projection']
    
                # Calculate 2D error for this view
                error_2d = np.linalg.norm(original_2d - reprojected_2d)
                reprojection_errors[camera_name] = error_2d
    
        return {
            'original': original_point,
            'reconstructed': reconstructed_point,
            'error_3d': error_3d,
            'reprojection_errors': reprojection_errors,
            'mean_reprojection_error': np.mean(list(reprojection_errors.values())),
            'num_views_used': len(valid_projections)
        }

def main():
    """Test the implementation with simplified triangulation"""
    setup = IsometricCameraSetup(
        radius=2.0,
        height=1.5,
        focal_length=500,
        image_size=(1000, 1000),
        subject_height=0.3
    )

    # Test points
    test_points = [
        np.array([-2, 0, 1.3]),          # Center
        np.array([2.1, 2, 0.3]),        # Slight right
        np.array([0, 0.1, 0.3]),        # Slight forward
        np.array([0.05, 1.05, 0.35]),    # Slight elevated
        np.array([0, 0, 1.3]),          # Center reference  
        # np.array([0.5, 0, 0.3]),        # Far right  
        # np.array([0, 0.5, 0.3]),        # Far forward  
        # np.array([0.3, 0.3, 0.3]),      # Large diagonal  
        # np.array([0, 0, 1.0]),          # High elevation  
        # np.array([0.2, -0.2, 0.1]),     # Low with negative Y  
        # np.array([-0.3, 0.3, 0.6]),     # Negative X, high Z  
        # np.array([0.4, 0.4, 0.8]),      # Large offset in all dimensions  
        # np.array([0.1, 0.1, 0.0]),      # Ground level  
        np.array([-0.4, -0.4, 0.4])     # Far negative diagonal
    ]

    # Project and verify with triangulation
    for i, point in enumerate(test_points):
        print(f"\nTest Point {i+1}: {point}")
        projections = setup.project_point_to_all_views(point)

        # Print projections
        for proj in projections:
            status = "(valid)" if proj['valid'] else "(outside)"
            print(f"{proj['camera']}: {proj['projection']} {status}")

        # Verify triangulation
        verification = setup.verify_triangulation(point, projections)
        print("\nTriangulation Results:")
        print(f"Original point:     {verification['original']}")
        print(f"Reconstructed point: {verification['reconstructed']}")
        print(f"Error:              {verification['error']:.6f}")
        print(f"Views used:         {verification['num_views_used']}")

    setup.visualize_camera_setup(test_points)
    




if __name__ == "__main__":
    main()