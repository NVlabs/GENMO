import os
import os.path as osp
import platform
import random
import subprocess

import cv2 as cv
import numpy as np
import seaborn as sns
from PIL import ImageColor

FFMPEG_PATH = "/usr/bin/ffmpeg" if osp.exists("/usr/bin/ffmpeg") else "ffmpeg"
font_files = {
    "Windows": "C:/Windows/Fonts/arial.ttf",
    "Linux": "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
    "Darwin": "/System/Library/Fonts/Supplemental/Arial.ttf",
}


def get_video_width_height(video_file):
    vcap = cv.VideoCapture(video_file)
    img_w = int(vcap.get(3))
    img_h = int(vcap.get(4))
    return img_w, img_h


def get_video_num_fr(video_file):
    vcap = cv.VideoCapture(video_file)
    num_fr = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))
    return num_fr


def get_video_fps(video_file):
    vcap = cv.VideoCapture(video_file)
    fps = vcap.get(cv.CAP_PROP_FPS)
    return fps


def rescale_video(video_path, out_path, width=-1, height=-1, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"scale={width}:{height}",
        out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)


def crop_video(
    video_path, out_path, width="iw", height="ih", x=0, y=0, crop_str=None, verbose=True
):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    if crop_str is None:
        crop_str = f"{width}:{height}:{x}:{y}"
    cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vf", f"crop={crop_str}", out_path]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)


def clip_video(video_path, out_path, start, end, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss",
        f"{start}",
        "-i",
        video_path,
        "-t",
        f"{end - start}",
        out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)


def text_video(
    video_path,
    out_path,
    text,
    x=10,
    y=30,
    verbose=True,
    text_color="white",
    text_size=60,
):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    font_file = font_files[platform.system()]
    draw_str = f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text}':x={x}:y={y}"
    cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vf", draw_str, out_path]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)


def make_gif(
    video_path,
    out_path,
    t_start="00:00:00",
    t_end=None,
    width=600,
    height=-1,
    verbose=True,
):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = (
        [FFMPEG_PATH, "-y", "-i", video_path, "-ss", t_start]
        + (["-t", t_end] if t_end is not None else [])
        + [
            "-vf",
            f"fps=30,scale={width}:{height}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            out_path,
        ]
    )
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)


def images_to_video(
    img_dir, out_path, img_fmt="%06d.jpg", fps=30, crf=25, verbose=True
):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-r",
        f"{fps}",
        "-f",
        "image2",
        "-start_number",
        "0",
        "-i",
        f"{img_dir}/{img_fmt}",
        "-vcodec",
        "libx264",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf",
        f"{crf}",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise Exception("Something went wrong during images_to_video!")


def video_to_images(video_path, out_path, img_fmt="%06d.jpg", fps=30, verbose=True):
    os.makedirs(out_path, exist_ok=True)
    cmd = [FFMPEG_PATH, "-i", video_path, "-r", f"{fps}", f"{out_path}/{img_fmt}"]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise Exception("Something went wrong during video_to_images!")


def hstack_videos(
    video1_path,
    video2_path,
    out_path,
    crf=25,
    verbose=True,
    text1=None,
    text2=None,
    text_color="white",
    text_size=60,
):
    if not (text1 is None or text2 is None):
        write_text = True
        tmp_file = f"{osp.splitext(out_path)[0]}_tmp.mp4"
    else:
        write_text = False

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        video1_path,
        "-i",
        video2_path,
        "-filter_complex",
        "hstack,format=yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        f"{crf}",
        tmp_file if write_text else out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)

    if write_text:
        font_file = font_files[platform.system()]
        draw_str = (
            f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text1}':x=(w-text_w)/4:y=20,"
            f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text2}':x=3*(w-text_w)/4:y=20"
        )
        cmd = [
            FFMPEG_PATH,
            "-i",
            tmp_file,
            "-y",
            "-vf",
            draw_str,
            "-c:a",
            "copy",
            out_path,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
        os.remove(tmp_file)


def vstack_videos(
    video1_path,
    video2_path,
    out_path,
    crf=25,
    verbose=True,
    text1=None,
    text2=None,
    text_color="white",
    text_size=60,
):
    if not (text1 is None or text2 is None):
        write_text = True
        tmp_file = f"{osp.splitext(out_path)[0]}_tmp.mp4"
    else:
        write_text = False

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        video1_path,
        "-i",
        video2_path,
        "-filter_complex",
        "vstack,format=yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        f"{crf}",
        tmp_file if write_text else out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)

    if write_text:
        font_file = font_files[platform.system()]
        draw_str = (
            f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text1}':x=10:y=20,"
            f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text2}':x=10:y=h/2+20"
        )
        cmd = [
            FFMPEG_PATH,
            "-i",
            tmp_file,
            "-y",
            "-vf",
            draw_str,
            "-c:a",
            "copy",
            out_path,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
        os.remove(tmp_file)


def vstack_video_arr(
    video_arr,
    out_path,
    crf=25,
    verbose=True,
    text_arr=None,
    text_color="white",
    text_size=60,
):
    assert len(video_arr) > 1
    tmp_file1 = f"{osp.splitext(out_path)[0]}_tmp1.mp4"
    tmp_file2 = f"{osp.splitext(out_path)[0]}_tmp2.mp4"

    height = np.array([get_video_width_height(x)[1] for x in video_arr])
    start_h = np.concatenate([np.array([0]), np.cumsum(height)[:-1]])

    os.makedirs(osp.dirname(out_path), exist_ok=True)

    for i in range(1, len(video_arr)):
        prev_video = video_arr[0] if i == 1 else tmp_file1
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            prev_video,
            "-i",
            video_arr[i],
            "-filter_complex",
            "vstack,format=yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            f"{crf}",
            tmp_file2,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
        tmp_file1, tmp_file2 = tmp_file2, tmp_file1

    if text_arr is not None:
        font_file = font_files[platform.system()]
        draw_str = ",".join(
            [
                f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{x}':x=10:y={h}+20"
                for h, x in zip(start_h, text_arr)
            ]
        )
        cmd = [
            FFMPEG_PATH,
            "-i",
            tmp_file1,
            "-y",
            "-vf",
            draw_str,
            "-c:a",
            "copy",
            out_path,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
    else:
        os.rename(tmp_file1, out_path)

    if os.path.exists(tmp_file1):
        os.remove(tmp_file1)
    if os.path.exists(tmp_file2):
        os.remove(tmp_file2)


def hstack_video_arr(
    video_arr,
    out_path,
    crf=25,
    verbose=True,
    text_arr=None,
    text_color="white",
    text_size=60,
):
    assert len(video_arr) > 1
    tmp_file1 = f"{osp.splitext(out_path)[0]}_tmp1.mp4"
    tmp_file2 = f"{osp.splitext(out_path)[0]}_tmp2.mp4"

    width = np.array([get_video_width_height(x)[0] for x in video_arr])
    start_w = np.concatenate([np.array([0]), np.cumsum(width)[:-1]])

    os.makedirs(osp.dirname(out_path), exist_ok=True)

    for i in range(1, len(video_arr)):
        prev_video = video_arr[0] if i == 1 else tmp_file1
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            prev_video,
            "-i",
            video_arr[i],
            "-filter_complex",
            "hstack,format=yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            f"{crf}",
            tmp_file2,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
        tmp_file1, tmp_file2 = tmp_file2, tmp_file1

    if text_arr is not None:
        font_file = font_files[platform.system()]
        draw_str = ",".join(
            [
                f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{x}':x={w}+10:y=20"
                for w, x in zip(start_w, text_arr)
            ]
        )
        cmd = [
            FFMPEG_PATH,
            "-i",
            tmp_file1,
            "-y",
            "-vf",
            draw_str,
            "-c:a",
            "copy",
            out_path,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
    else:
        os.rename(tmp_file1, out_path)

    if os.path.exists(tmp_file1):
        os.remove(tmp_file1)
    if os.path.exists(tmp_file2):
        os.remove(tmp_file2)


def make_checker_board_texture(
    color1="black", color2="white", width=10, height=10, n_tile=15, to_bgr=False
):
    c1 = np.asarray(ImageColor.getcolor(color1, "RGB")).astype(np.uint8)
    c2 = np.asarray(ImageColor.getcolor(color2, "RGB")).astype(np.uint8)
    if to_bgr:
        c1 = c1[[2, 1, 0]]
        c2 = c2[[2, 1, 0]]
    hw = width // 2
    hh = height // 2
    c1_block = np.tile(c1, (hh, hw, 1))
    c2_block = np.tile(c2, (hh, hw, 1))
    tex = np.block([[[c1_block], [c2_block]], [[c2_block], [c1_block]]])
    tex = np.tile(tex, (n_tile, n_tile, 1))
    return tex


def resize_bbox(bbox, scale):
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    h, w = y2 - y1, x2 - x1
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
    h_new, w_new = h * scale, w * scale
    x1_new, x2_new = cx - 0.5 * w_new, cx + 0.5 * w_new
    y1_new, y2_new = cy - 0.5 * h_new, cy + 0.5 * h_new
    bbox_new = np.stack([x1_new, y1_new, x2_new, y2_new], axis=-1)
    return bbox_new


def nparray_to_vtk_matrix(array):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4"""
    import vtk

    matrix = vtk.vtkMatrix4x4()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])
    return matrix


def vtk_matrix_to_nparray(matrix):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4"""
    array = np.zeros([4, 4])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.GetElement(i, j)
    return array


def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def draw_tracks(
    img, bbox, idx, score, thickness=2, font_scale=0.4, text_height=10, text_width=15
):
    # taken from mmtracking
    x1, y1, x2, y2 = bbox.astype(np.int32)

    # bbox
    bbox_color = random_color(idx)
    bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
    cv.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

    # id
    text = str(idx)
    width = len(text) * text_width
    img[y1 : y1 + text_height, x1 : x1 + width, :] = bbox_color
    cv.putText(
        img,
        str(idx),
        (x1, y1 + text_height - 2),
        cv.FONT_HERSHEY_COMPLEX,
        font_scale,
        color=(0, 0, 0),
    )

    # score
    text = "{:.02f}".format(score)
    width = len(text) * text_width
    img[y1 - text_height : y1, x1 : x1 + width, :] = bbox_color
    cv.putText(
        img, text, (x1, y1 - 2), cv.FONT_HERSHEY_COMPLEX, font_scale, color=(0, 0, 0)
    )
    return img


def draw_keypoints(img, keypoints, confidence, size=4, color=(255, 0, 255)):
    for kp, conf in zip(keypoints, confidence):
        if conf > 0.2:
            cv.circle(
                img, np.round(kp).astype(int).tolist(), size, color=color, thickness=-1
            )
    return img
