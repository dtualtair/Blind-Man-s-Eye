"""
Blind Man's Eye (BME) - Utility Functions for Monocular Depth
=============================================================

This module provides helper utilities used across the BME depth estimation pipeline:
- Read/write PFM files (portable float map) for high-precision depth data
- Read images as normalized RGB arrays
- Resize images/depths to meet network constraints (multiples of 32, etc.)
- Write depth maps to PNG in 8-bit color or 16-bit grayscale

Notes:
- Depth tensors/arrays are handled carefully to avoid NaNs/Infs in outputs.
- PNG output supports 8-bit (colorized) or 16-bit (grayscale) depth encodings.
- PFM read/write preserves float32 precision for scientific processing.
"""

import sys
import re
import numpy as np
import cv2
import torch


def read_pfm(path: str):
    """
    Read a PFM (Portable Float Map) file and return the data with its scale.

    Args:
        path (str): Path to the PFM file.

    Returns:
        tuple[np.ndarray, float]: (image_data, scale), where:
            - image_data: float32 numpy array, shape (H, W) for grayscale or (H, W, 3) for color
            - scale: scale factor read from file (negative indicates little-endian)

    Raises:
        Exception: If the file header is not a valid PFM header or is malformed.

    PFM format recap:
    - First line: "PF" (color) or "Pf" (grayscale)
    - Second line: "<width> <height>"
    - Third line: scale (negative for little-endian, positive for big-endian)
    - Then raw float32 data (row-major, top scanline last; hence we flip vertically)
    """
    with open(path, "rb") as file:
        # Parse header: "PF" for color, "Pf" for grayscale
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        # Parse dimensions: width height
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        # Parse scale and endianness
        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # Negative scale indicates little-endian
            endian = "<"
            scale = -scale
        else:
            # Positive scale indicates big-endian
            endian = ">"

        # Read the actual image data as float32
        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        # Reshape and vertically flip (PFM stores image bottom-to-top)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data.astype("float32"), scale


def write_pfm(path: str, image: np.ndarray, scale: float = 1.0):
    """
    Write a numpy float32 array to a PFM file.

    Args:
        path (str): Output file path (without enforced extension).
        image (np.ndarray): Image data as float32; shape:
                            - H x W for grayscale
                            - H x W x 3 for color
        scale (float, optional): Scale factor. Defaults to 1.0.
                                 Negative scale will be written for little-endian systems.

    Raises:
        Exception: If dtype is not float32 or shape is invalid.

    Notes:
        - PFM format expects float32 data and stores rows from bottom to top.
        - Endianness is indicated via the sign of the scale value.
    """
    with open(path, "wb") as file:
        # Validate dtype is float32 as per PFM spec
        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        # Flip image vertically to comply with PFM storage order
        image = np.flipud(image)

        # Determine if image is color (H x W x 3) or grayscale (H x W or H x W x 1)
        if len(image.shape) == 3 and image.shape[2] == 3:
            color = True
        elif (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1):
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        # Write header: "PF\n" or "Pf\n"
        file.write(("PF\n" if color else "Pf\n").encode())

        # Write dimensions
        file.write(f"{image.shape[1]} {image.shape[0]}\n".encode())

        # Endianness handling: negative scale for little-endian
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale

        # Write scale
        file.write(f"{scale}\n".encode())

        # Write raw float32 data
        image.tofile(file)


def read_image(path: str) -> np.ndarray:
    """
    Read an image from disk and return an RGB image normalized to [0, 1].

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: RGB float image with values in [0, 1], shape (H, W, 3).

    Notes:
        - Converts grayscale to BGR first, then to RGB.
        - Uses OpenCV for I/O; OpenCV reads images as BGR by default.
    """
    img = cv2.imread(path)

    # If grayscale, convert to 3-channel BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert BGR -> RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype("float32")


def resize_image(img: np.ndarray) -> torch.Tensor:
    """
    Resize an RGB image so that the larger dimension scales to ~384, while preserving aspect
    ratio and aligning the result to multiples of 32 (as many encoders require).

    Args:
        img (np.ndarray): RGB image in [0, 1], shape (H, W, 3).

    Returns:
        torch.Tensor: Image tensor of shape (1, 3, H_aligned, W_aligned), dtype float32.

    Notes:
        - Uses INTER_AREA for downscaling.
        - Returns a batched CHW tensor (NCHW) suitable for PyTorch models.
    """
    height_orig, width_orig = img.shape[0], img.shape[1]

    # Decide scaling factor so that the larger side maps roughly to 384
    scale = (width_orig / 384) if (width_orig > height_orig) else (height_orig / 384)

    # Compute target size and align to multiples of 32
    height = int(np.ceil(height_orig / scale / 32) * 32)
    width = int(np.ceil(width_orig / scale / 32) * 32)

    # Resize with area interpolation for downsampling quality
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # HWC -> CHW and convert to contiguous float tensor
    img_resized = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()

    # Add batch dimension: (1, 3, H, W)
    return img_resized.unsqueeze(0)


def resize_depth(depth: torch.Tensor, width: int, height: int) -> np.ndarray:
    """
    Resize a depth tensor to the target (width, height) and return as a numpy array (CPU).

    Args:
        depth (torch.Tensor): Depth prediction tensor of shape (N, C, H, W) or (1, 1, H, W).
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Resized depth map as float32 array of shape (height, width).

    Notes:
        - Squeezes batch/channel dims before resizing.
        - Uses bicubic interpolation for smoother depth transitions.
    """
    # Squeeze to (H, W) and move to CPU
    depth_2d = torch.squeeze(depth[0, :, :, :]).to("cpu")

    # Resize to target resolution
    depth_resized = cv2.resize(depth_2d.numpy(), (width, height), interpolation=cv2.INTER_CUBIC)
    return depth_resized.astype("float32")


def write_depth(path: str, depth: np.ndarray, grayscale: bool, bits: int = 1):
    """
    Write a depth map to a PNG file with optional colorization or 16-bit grayscale.

    Args:
        path (str): File path without extension (PNG will be appended).
        depth (np.ndarray): Depth map as float array.
        grayscale (bool): If True, write in grayscale; otherwise colorize with 'inferno'.
        bits (int): Bit depth for PNG when grayscale=True:
                    - If grayscale=False, this function forces bits=1 (8-bit color).
                    - If grayscale=True and bits=2, writes 16-bit grayscale PNG.

    Returns:
        None

    Behavior:
        - Non-finite values (NaN/Inf) are replaced with zeros to ensure valid PNGs.
        - Depth is normalized to [0, max_val] where max_val depends on bit depth.
        - Colormap 'inferno' is applied when grayscale=False (8-bit output).
    """
    # Enforce 8-bit output when using color colormap
    if not grayscale:
        bits = 1

    # Replace NaN/Inf with 0 to prevent PNG write errors
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = float(depth.min())
    depth_max = float(depth.max())

    # Compute maximum value for the selected bit depth
    max_val = (2 ** (8 * bits)) - 1

    # Normalize depth to [0, max_val]
    if depth_max - depth_min > np.finfo("float32").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    # Apply colormap if not grayscale
    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    # Write appropriate PNG bit depth
    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return
