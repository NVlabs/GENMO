import os
import shutil
from pathlib import Path

import cv2
import ffmpeg
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm


def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(
                ("trim", f"start_frame={start_frame}:end_frame={end_frame}")
            )

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [
            cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)
        ]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """remember to .close()"""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def concat_videos(cfg, out_video_path: str, in_video_paths=None):
    # if len(in_video_paths) < 2:
    #     raise ValueError("At least two video paths are required for merging.")
    # in_video_paths = [cfg.video1_path, cfg.text1_video_path, cfg.video2_path]
    if in_video_paths is None:
        in_video_paths = [
            cfg.paths.incam_video1,
            cfg.text1_video_path,
            cfg.paths.incam_video2,
        ]

    # Get the size of the first video to use as target size
    probe = ffmpeg.probe(in_video_paths[0])
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    target_size = (int(video_stream["width"]), int(video_stream["height"]))

    # Resize and pad all videos to match the target size
    temp_paths = [resize_and_pad_video(path, target_size) for path in in_video_paths]

    try:
        # Create inputs from the resized videos
        inputs = [ffmpeg.input(path) for path in temp_paths]
        merged_video = ffmpeg.concat(*inputs)
        output = ffmpeg.output(merged_video, out_video_path)
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)


def resize_and_pad_video(video_path, target_size):
    """
    Resize and pad a video to match the target size.

    Args:
        video_path: Path to the input video
        target_size: Tuple of (width, height) for the target size

    Returns:
        Path to the resized and padded temporary video
    """
    import os
    import tempfile

    target_width, target_height = target_size

    # Create a temporary file for the resized video
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Get video info
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Calculate scaling to maintain aspect ratio
    if width / height > target_width / target_height:
        # Width is the limiting factor
        scale_w = target_width
        scale_h = -1  # Maintain aspect ratio
    else:
        # Height is the limiting factor
        scale_w = -1  # Maintain aspect ratio
        scale_h = target_height

    # Resize and pad
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, "scale", scale_w, scale_h)
    stream = ffmpeg.filter(
        stream, "pad", target_width, target_height, "(ow-iw)/2", "(oh-ih)/2"
    )
    stream = ffmpeg.output(stream, temp_path)
    ffmpeg.run(stream, quiet=True, overwrite_output=True)

    return temp_path


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
