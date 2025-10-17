"""
Blind Man's Eye (BME) - Depth Map Batch/Camera Runner
=====================================================

This script computes depth maps using MiDaS/DPT models for:
- All images in an input folder (batch mode), or
- A live camera stream (real-time mode when no input folder is provided)

It supports multiple MiDaS model variants (DPT/BEiT/Swin/ViT/LeViT) and includes:
- Optional half-float optimization on CUDA
- Side-by-side RGB + depth visualization
- Grayscale or colored depth outputs
- FPS display in camera mode
- OpenVINO path (if model_type contains "openvino")

Project: Blind Man's Eye (BME)
License: MIT
"""

import os
import glob
import torch
import utils
import cv2
import argparse
import time
import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

# Used to print one-time info (e.g., resized input) only on first pass
first_execution = True


def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate to the target resolution.

    Args:
        device (torch.device): Torch device (cpu/cuda) used for inference
        model: Loaded MiDaS/DPT model or OpenVINO model
        model_type (str): Model type key/name (e.g., 'dpt_hybrid_384', 'openvino_midas_v21_small_256')
        image (np.ndarray or torch.Tensor): Preprocessed input image (CHW, float)
        input_size (Tuple[int, int]): (width, height) the network input size (for OpenVINO path)
        target_size (Tuple[int, int]): (width, height) target output resolution (original image size)
        optimize (bool): Use half-float optimization on CUDA (where appropriate)
        use_camera (bool): True if frames are coming from camera (affects once-per-run prints)

    Returns:
        np.ndarray: Predicted depth map (H, W) as float32 numpy array
    """
    global first_execution

    # OpenVINO inference path (expects numpy inputs shaped for the runtime)
    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        # OpenVINO expects NHWC-like numpy input depending on conversion;
        # here we reshape to the expected (1, 3, H, W) as prepared upstream.
        sample = [np.reshape(image, (1, 3, *input_size))]

        # Forward pass through OpenVINO model; obtain first output tensor
        prediction = model(sample)[model.output(0)][0]

        # Resize prediction to the original target size (W, H)
        prediction = cv2.resize(prediction, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    else:
        # PyTorch path: convert numpy (CHW) to torch tensor and add batch
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        # Optional half-float optimization on CUDA (use with caution for some backbones)
        if optimize and device == torch.device("cuda"):
            if first_execution:
                print(
                    "  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats."
                )
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        # Print input shape once (or when not using camera)
        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        # Forward pass
        prediction = model.forward(sample)

        # Interpolate to target size (note: target_size is (W, H), interpolate needs (H, W))
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),          # add channel dimension
                size=target_size[::-1],           # reverse to (H, W)
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Concatenate RGB image and normalized depth map side by side for visualization.

    Args:
        image (np.ndarray): Input BGR or RGB image (H, W, 3), values typically [0, 255]
        depth (np.ndarray): Depth map (H, W), float
        grayscale (bool): If True, produce grayscale; else apply inferno colormap

    Returns:
        np.ndarray: Side-by-side visualization (H, 2*W, 3)
    """
    # Normalize depth for display (0-255) with simple contrast scaling
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min + 1e-8)
    normalized_depth *= 3  # boost contrast a bit

    # Make 3-channel depth visualization
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    # If no image (e.g., camera-off/side-only), just return the depth map visualization
    if image is None:
        return right_side
    else:
        # Ensure image is uint8 for proper concatenation
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.concatenate((image, right_side), axis=1)


def run(
    input_path,
    output_path,
    model_path,
    model_type="dpt_beit_large_512",
    optimize=False,
    side=False,
    height=None,
    square=False,
    grayscale=False,
):
    """
    Run BME depth computation for a folder of images or a camera stream.

    Args:
        input_path (str): Path to input images folder; if None, use camera stream
        output_path (str): Path to save outputs; if None, don't save
        model_path (str): Path/URL to model weights (or key in default_models)
        model_type (str): Model type key (see CLI help for choices)
        optimize (bool): Use half-float optimization on CUDA
        side (bool): Output side-by-side RGB+Depth if True; else write depth-only
        height (int): Preferred encoder input height (some models ignore this)
        square (bool): Force square input (width adapted) if True
        grayscale (bool): Use grayscale depth visualization instead of inferno
    """
    print("Initialize")

    # Select compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load model, pre-processing transform, and network input size
    # net_w, net_h are the input width/height that the model expects
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # Gather inputs
    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)
    else:
        print("No input path specified. Grabbing images from camera.")

    # Prepare output directory (if saving)
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        # Batch image mode
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):
            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

            # Load and preprocess input image (RGB in [0,1])
            original_image_rgb = utils.read_image(image_name)
            image = transform({"image": original_image_rgb})["image"]

            # Inference
            with torch.no_grad():
                prediction = process(
                    device,
                    model,
                    model_type,
                    image,
                    (net_w, net_h),
                    original_image_rgb.shape[1::-1],  # (W, H)
                    optimize,
                    use_camera=False,
                )

            # Output handling
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + "-" + model_type
                )

                if not side:
                    # Write depth PNG (16-bit if grayscale=True -> utils decides bits)
                    utils.write_depth(filename, prediction, grayscale, bits=2)
                else:
                    # Side-by-side visualization
                    original_image_bgr = np.flip(original_image_rgb, 2)  # RGB->BGR for OpenCV show/write
                    content = create_side_by_side(original_image_bgr * 255, prediction, grayscale)
                    cv2.imwrite(filename + ".png", content)

                # Also write PFM (float) for precise depth values
                utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))

    else:
        # Camera stream mode
        with torch.no_grad():
            fps = 1.0
            video = VideoStream(0).start()  # Start camera
            time_start = time.time()
            frame_index = 0

            while True:
                frame = video.read()
                if frame is not None:
                    # Convert BGR->RGB; keep in [0,255]
                    original_image_rgb = np.flip(frame, 2)  # OpenCV BGR to RGB
                    # Normalize to [0,1] for transform
                    image = transform({"image": original_image_rgb / 255.0})["image"]

                    # Run inference
                    prediction = process(
                        device,
                        model,
                        model_type,
                        image,
                        (net_w, net_h),
                        original_image_rgb.shape[1::-1],  # (W, H)
                        optimize,
                        use_camera=True,
                    )

                    # If side-by-side requested, show RGB+Depth; else show Depth-only
                    original_image_bgr = np.flip(original_image_rgb, 2) if side else None
                    content = create_side_by_side(original_image_bgr, prediction, grayscale)

                    # Display scaled to [0,1] window if content is uint8-like; divide by 255 for float display
                    cv2.imshow("MiDaS Depth Estimation - Press Escape to close window ", content / 255.0)

                    # Optionally save frames
                    if output_path is not None:
                        filename = os.path.join(output_path, "Camera" + "-" + model_type + "_" + str(frame_index))
                        cv2.imwrite(filename + ".png", content)

                    # FPS smoothing (exponential moving average)
                    alpha = 0.1
                    if time.time() - time_start > 0:
                        fps = (1 - alpha) * fps + alpha * 1 / (time.time() - time_start)
                        time_start = time.time()
                    print(f"\rFPS: {round(fps, 2)}", end="")

                    # Exit on ESC
                    if cv2.waitKey(1) == 27:  # Escape key
                        break

                    frame_index += 1

        print()  # newline for clean prompt

    print("Finished")


if __name__ == "__main__":
    # Command-line interface for batch/camera processing
    parser = argparse.ArgumentParser(description="Blind Man's Eye (BME) - Depth Estimation Runner")

    parser.add_argument(
        "-i",
        "--input_path",
        default=None,
        help=(
            "Folder with input images (if no input path is specified, frames are grabbed from camera). "
            "Example: -i ./input_images"
        ),
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default=None,
        help="Folder for output images. Example: -o ./outputs",
    )

    parser.add_argument(
        "-m",
        "--model_weights",
        default=None,
        help="Path or URL to the trained model weights (if omitted, a default will be selected based on model_type).",
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_beit_large_512",
        help=(
            "Model type: "
            "dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, "
            "dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, "
            "dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or "
            "openvino_midas_v21_small_256"
        ),
    )

    parser.add_argument(
        "-s",
        "--side",
        action="store_true",
        help="Output images contain RGB and depth images side by side (otherwise depth-only).",
    )

    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        help="Use half-float (FP16) optimization on CUDA (use with caution for Swin-type models).",
    )
    parser.set_defaults(optimize=False)

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Preferred height fed into the encoder during inference. "
            "May be adjusted to multiples of 32. Some models only support their training height."
        ),
    )

    parser.add_argument(
        "--square",
        action="store_true",
        help=(
            "Resize images to a square resolution by adapting widths during inference. "
            "If unset, aspect ratio is preserved when supported by the model."
        ),
    )

    parser.add_argument(
        "--grayscale",
        action="store_true",
        help=(
            "Use a grayscale colormap instead of inferno. Note: inferno provides better visibility, "
            "but grayscale allows storing 16-bit depth in PNG when bits=2 in write_depth."
        ),
    )

    args = parser.parse_args()

    # Auto-select weights if not provided
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # Enable cuDNN autotuner for performance on variable input sizes
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Compute depth maps (batch or camera mode)
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
        args.side,
        args.height,
        args.square,
        args.grayscale,
    )
