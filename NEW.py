"""
Blind Man's Eye (BME) - Main Depth Estimation Module
====================================================

This module implements real-time monocular depth estimation using Intel's MiDaS model
for the BME (Blind Man's Eye) assistive navigation system.

License: MIT

Features:
- Real-time depth estimation using MiDaS v2.1/v3
- Color-coded depth visualization (MAGMA colormap)
- Obstacle detection through HSV color filtering
- Real-time camera processing with FPS monitoring
- GPU acceleration support
"""

import cv2
import torch
import time
import numpy as np
import sys
import logging
from typing import Tuple, List

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BMEDepthEstimator:
    """
    Blind Man's Eye Depth Estimation System

    This class handles real-time monocular depth estimation and obstacle detection
    for visually impaired navigation assistance.
    """

    def __init__(self, model_type: str = "MiDaS_small"):
        """
        Initialize the BME Depth Estimator

        Args:
            model_type (str): MiDaS model variant to use
                              Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        """
        self.model_type = model_type
        self.device = None
        self.midas = None
        self.transform = None
        self.cap = None

        # Performance monitoring
        self.frame_count = 0
        self.total_time = 0
        self.avg_fps = 0

        # Obstacle detection parameters (red in MAGMA colormap ~ close)
        self.red_lower = np.array([0, 100, 60], np.uint8)     # Lower HSV threshold for "close" regions
        self.red_upper = np.array([35, 255, 255], np.uint8)   # Upper HSV threshold
        self.kernel_size = (5, 5)                             # Morphological operations kernel
        self.min_contour_area = 100                           # Minimum contour area to count as obstacle

        # Display parameters
        self.display_width = 192 * 3    # 576px
        self.display_height = 108 * 4   # 432px

        logger.info(f"Initializing BME Depth Estimator with model: {model_type}")
        self._initialize_model()
        self._initialize_camera()

    def _initialize_model(self) -> None:
        """
        Initialize the MiDaS depth estimation model.

        Sets up the appropriate model variant and transformations based on model_type.
        Automatically detects and configures GPU acceleration if available.
        """
        try:
            logger.info("Loading MiDaS model from PyTorch Hub...")

            # Load the specified MiDaS model from Intel's repository
            self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

            # Configure compute device (GPU preferred for better performance)
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            logger.info(f"Using device: {self.device}")

            # Move model to device and set to evaluation mode
            self.midas.to(self.device)
            self.midas.eval()

            # Load appropriate image transformations for the model
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            # Select transformation based on model architecture
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
                logger.info("Using DPT transformation pipeline")
            else:
                self.transform = midas_transforms.small_transform
                logger.info("Using small model transformation pipeline")

            logger.info("Model initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            sys.exit(1)

    def _initialize_camera(self) -> None:
        """
        Initialize camera capture for real-time processing.

        Sets up OpenCV VideoCapture with error handling for camera access issues.
        """
        try:
            logger.info("Initializing camera capture...")
            self.cap = cv2.VideoCapture(0)  # Default camera

            if not self.cap.isOpened():
                raise ValueError("Cannot access camera. Please check camera connection.")

            # Optional: set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Camera initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            sys.exit(1)

    def _process_frame(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single frame for depth estimation.

        Args:
            img (np.ndarray): Input BGR image from camera

        Returns:
            Tuple[np.ndarray, float]: Normalized depth map [0,1] and inference time in seconds
        """
        start_time = time.time()

        # Convert BGR to RGB (MiDaS expects RGB input)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply model-specific transformations (resize, normalize, etc.)
        input_batch = self.transform(img_rgb).to(self.device)

        # Perform depth estimation inference
        with torch.no_grad():  # No gradients needed for inference
            prediction = self.midas(input_batch)

            # Interpolate prediction back to original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),            # Add channel dimension
                size=img_rgb.shape[:2],             # Target HxW
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()

        # Normalize to [0, 1] for consistent visualization
        depth_map = cv2.normalize(
            depth_map,
            None,
            0, 1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_64F
        )

        processing_time = time.time() - start_time
        return depth_map, processing_time

    def _detect_obstacles(self, depth_map: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect obstacles in the depth map using color-based segmentation.

        The MiDaS depth map is colorized using the MAGMA colormap where red regions
        indicate closer objects (potential obstacles).

        Args:
            depth_map (np.ndarray): Normalized depth map [0, 1]

        Returns:
            Tuple[np.ndarray, List[dict]]: Color-coded depth map (BGR) and obstacle metadata list
        """
        # Convert normalized depth map to 8-bit
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)

        # Apply MAGMA colormap for visualization
        colored_depth = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_MAGMA)

        # Convert to HSV for color-based segmentation
        hsv_frame = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2HSV)

        # Mask for red/orange/yellow regions (close in MAGMA)
        red_mask = cv2.inRange(hsv_frame, self.red_lower, self.red_upper)

        # Morphological cleanup
        kernel = np.ones(self.kernel_size, np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)

        # Find contours for potential obstacles
        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract obstacle metadata
        obstacles = []
        h, w = colored_depth.shape[:2]
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Estimate relative distance using mask intensity (heuristic)
            y0, y1 = max(cy - 10, 0), min(cy + 10, h)
            x0, x1 = max(cx - 10, 0), min(cx + 10, w)
            region = red_mask[y0:y1, x0:x1]
            avg_intensity = float(np.mean(region)) if region.size else 0.0
            relative_distance = 1.0 - (avg_intensity / 255.0)  # Higher intensity => closer => lower value

            # Relative direction: -1 (left) to +1 (right)
            relative_direction = (cx - (w / 2)) / (w / 2)

            obstacles.append({
                "id": i,
                "centroid": (cx, cy),
                "area": area,
                "relative_distance": relative_distance,
                "relative_direction": relative_direction,
                "contour": contour
            })

        return colored_depth, obstacles

    def _draw_obstacles(self, img: np.ndarray, obstacles: List[dict]) -> np.ndarray:
        """
        Draw obstacle indicators on the image.

        Args:
            img (np.ndarray): Input image (BGR)
            obstacles (List[dict]): Detected obstacle metadata

        Returns:
            np.ndarray: Annotated image
        """
        annotated = img.copy()

        for obs in obstacles:
            cx, cy = obs["centroid"]
            rel_dist = obs["relative_distance"]
            rel_dir = obs["relative_direction"]

            # Draw contour and centroid
            cv2.drawContours(annotated, [obs["contour"]], -1, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)

            # Human-readable labels
            direction = "LEFT" if rel_dir < -0.3 else "RIGHT" if rel_dir > 0.3 else "CENTER"
            distance = "NEAR" if rel_dist < 0.3 else "MID" if rel_dist < 0.7 else "FAR"
            label = f"{direction}-{distance}"

            cv2.putText(
                annotated, label, (cx - 50, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return annotated

    def _update_performance_stats(self, processing_time: float) -> None:
        """
        Update performance statistics for monitoring.

        Args:
            processing_time (float): Time taken for current frame processing
        """
        self.frame_count += 1
        self.total_time += processing_time

        if self.frame_count % 10 == 0:
            self.avg_fps = 10.0 / (self.total_time if self.total_time > 0 else 1.0)
            self.total_time = 0.0
            logger.info(f"Average FPS (last 10 frames): {self.avg_fps:.1f}")

    def run_real_time_detection(self) -> None:
        """
        Main loop for real-time depth estimation and obstacle detection.

        Captures frames from camera, processes them for depth estimation,
        detects obstacles, and displays results in real-time.
        """
        logger.info("Starting real-time depth estimation...")
        logger.info("Press 'ESC' to quit, 'SPACE' to pause/resume, 'S' to save frame pair")

        paused = False

        try:
            while self.cap.isOpened():
                if not paused:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to capture frame from camera")
                        continue

                    # Depth estimation
                    depth_map, processing_time = self._process_frame(frame)

                    # Obstacle detection
                    colored_depth, obstacles = self._detect_obstacles(depth_map)

                    # Annotate original frame
                    annotated = self._draw_obstacles(frame, obstacles)

                    # Resize for display
                    display_frame = cv2.resize(
                        annotated, (self.display_width, self.display_height),
                        interpolation=cv2.INTER_AREA
                    )

                    # Show FPS and obstacle count
                    fps = 1.0 / processing_time if processing_time > 0 else 0.0
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Obstacles: {len(obstacles)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display windows
                    cv2.imshow("BME - Camera View", display_frame)
                    cv2.imshow("BME - Depth Map", colored_depth)

                    # Update performance stats
                    self._update_performance_stats(processing_time)

                    # TODO: Integrate spatial audio cues here (ITD/ILD-based)
                    # self._generate_audio_cues(obstacles)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("User requested quit")
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"bme_frame_{timestamp}.jpg", frame)
                    cv2.imwrite(f"bme_depth_{timestamp}.jpg", colored_depth)
                    logger.info(f"Saved frames with timestamp {timestamp}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """
        Clean up resources and close windows.
        """
        logger.info("Cleaning up resources...")

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        logger.info(f"Session completed. Total frames processed: {self.frame_count}")

    def process_image_file(self, image_path: str, output_path: str = None) -> None:
        """
        Process a single image file for depth estimation.

        Args:
            image_path (str): Path to input image
            output_path (str, optional): Base path to save outputs; if None, display instead
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")

            logger.info(f"Processing image: {image_path}")

            depth_map, _ = self._process_frame(img)
            colored_depth, obstacles = self._detect_obstacles(depth_map)
            annotated = self._draw_obstacles(img, obstacles)

            logger.info(f"Detected {len(obstacles)} obstacles")

            if output_path:
                cv2.imwrite(output_path.replace(".jpg", "_original.jpg"), img)
                cv2.imwrite(output_path.replace(".jpg", "_depth.jpg"), colored_depth)
                cv2.imwrite(output_path.replace(".jpg", "_annotated.jpg"), annotated)
                logger.info(f"Results saved to {output_path}")
            else:
                cv2.imshow("Original Image", img)
                cv2.imshow("Depth Map", colored_depth)
                cv2.imshow("Annotated Image", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")


def main():
    """
    Main function to run the BME Depth Estimation System.
    """
    print("=" * 60)
    print("Blind Man's Eye (BME) - Depth Estimation System")
    print("Team DTU Altair - HackVortex 2025")
    print("=" * 60)

    # Available model configurations
    MODEL_OPTIONS = {
        1: "MiDaS_small",   # Fastest, lowest accuracy
        2: "DPT_Hybrid",    # Medium speed and accuracy
        3: "DPT_Large"      # Slowest, highest accuracy
    }

    print("\nAvailable Models:")
    for key, model in MODEL_OPTIONS.items():
        print(f"{key}. {model}")

    try:
        choice = input("\nSelect model (1-3, default=1): ").strip()
        model_choice = int(choice) if choice.isdigit() and int(choice) in MODEL_OPTIONS else 1
        selected_model = MODEL_OPTIONS[model_choice]

        print(f"\nInitializing BME with {selected_model} model...")

        # Create and run BME system
        bme = BMEDepthEstimator(model_type=selected_model)

        # Run real-time detection
        bme.run_real_time_detection()

    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

