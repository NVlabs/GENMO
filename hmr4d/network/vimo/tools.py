import numpy as np


def parse_chunks(frame, boxes, min_len=16):
    """If a track disappear in the middle,
    we separate it to different segments to estimate the HPS independently.
    If a segment is less than 16 frames, we get rid of it for now.
    """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk) >= min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk == breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk) >= min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks
