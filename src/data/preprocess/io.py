"""I/O and file/video loading utilities.

This module collects input/output helpers used by the preprocessing pipeline.
Its main capabilities are:

* Checking whether a sample has already been processed
  (:func:`check_file_processed`).
* Computing sleep-stage durations from NSRR annotations (XML/CSV)
  (:func:`sum_stage_wake_duration`, :func:`sum_stage_wake_duration_from_csv`).
* Deriving the corresponding XML annotation path from an EDF path
  (:func:`get_xml_path_from_edf`).
* Loading and resizing RGB video/frames
  (:func:`loadVideo`, :func:`loadVideoStream`, :func:`loadVideoStreamCV2`,
  :func:`FrameResize`, :func:`loadFrames`).
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import imageio.v3 as iio
import cv2
import skimage.transform as sktransform
import itertools

def check_file_processed(base_name, save_path):
    """Check the existence and status of a sample's preprocessing outputs.

    Looks under ``save_path`` for ``{base_name}_hw.npy`` (heartbeat) and
    ``{base_name}_bw.npy`` (breathing) files. When the ``hw`` file exists, its
    size is also inspected to judge whether it was written successfully.

    Args:
        base_name (str): Base file name of the sample, without extension.
        save_path (str): Directory where the preprocessed ``.npy`` files are
            stored.

    Returns:
        tuple[bool, str]: A ``(is_complete, status)`` tuple. The status string
        is one of:

        * ``"complete"``: the ``hw`` file exists and has a valid size
          (processing complete).
        * ``"corrupted"``: the ``hw`` file exists but is too small, so it is
          treated as corrupted/incomplete.
        * ``"partial"``: only the ``bw`` file exists, indicating partial
          processing.
        * ``"new"``: neither file exists, indicating an unprocessed sample.
    """
    hw_file = os.path.join(save_path, base_name + '_hw.npy')
    bw_file = os.path.join(save_path, base_name + '_bw.npy')

    hw_exists = os.path.exists(hw_file)
    bw_exists = os.path.exists(bw_file)

    # Treat the sample as fully processed only when the hw file is present.
    if hw_exists:
        # Verify successful processing by checking the file size.
        hw_size = os.path.getsize(hw_file)

        if hw_size > 100:  # Minimum valid data size, in bytes.
            return True, "complete"
        else:
            return False, "corrupted"  # File exists but is corrupted or incomplete.

    # Only partially processed.
    elif bw_exists:
        return False, "partial"

    # Not processed yet.
    else:
        return False, "new"

def sum_stage_wake_duration(xml_path):
    """Sum the total duration of sleep-stage events from an NSRR XML annotation.

    Traverses the XML tree and accumulates the ``Duration`` value of every
    ``ScoredEvent`` whose ``EventType`` contains ``"Stages"``. Events whose
    ``Duration`` cannot be parsed as a number are ignored.

    Args:
        xml_path (str): Path to an NSRR-format annotation XML file.

    Returns:
        float: Total duration of the sleep-stage events, in seconds.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_duration = 0.0
    for scored_event in root.findall(".//ScoredEvent"):
        event_type = scored_event.find("EventType")
        if event_type is not None and event_type.text is not None and "Stages" in event_type.text:
            duration_elem = scored_event.find("Duration")
            if duration_elem is not None:
                try:
                    total_duration += float(duration_elem.text)
                except ValueError:
                    pass  # Ignore non-numeric values.
    return total_duration

def get_xml_path_from_edf(edf_path, xml_dir_shhs, xml_dir_mesa, dataset_type='SHHS'):
    """Derive the corresponding NSRR XML annotation path from an EDF path.

    Extracts the base name from the EDF file name to build an annotation file
    name of the form ``{base}-nsrr.xml``. The SHHS dataset assumes the XML
    directory mirrors the subdirectory structure (``shhs1``, ``shhs2``) in
    which the EDF resides, so that subdirectory is included in the path. The
    MESA dataset uses a single flat XML directory.

    Args:
        edf_path (str): Path to the source EDF file.
        xml_dir_shhs (str): Root directory of the SHHS XML annotations.
        xml_dir_mesa (str): Directory of the MESA XML annotations.
        dataset_type (str, optional): Dataset type (``'SHHS'`` or ``'MESA'``).
            Case-insensitive. Defaults to ``'SHHS'``.

    Returns:
        str: Full path to the corresponding XML annotation file.
    """
    base = os.path.splitext(os.path.basename(edf_path))[0]
    xml_filename = f"{base}-nsrr.xml"

    if dataset_type.upper() == 'SHHS':
        # Extract the subdirectory (shhs1, shhs2) from edf_path.
        edf_dir = os.path.dirname(edf_path)
        # Take the last directory name (shhs1 or shhs2).
        subdir = os.path.basename(edf_dir)

        # The XML directory mirrors the same subdirectory structure.
        xml_dir = os.path.join(xml_dir_shhs, subdir)
    else:  # MESA
        # XML path for the MESA dataset.
        xml_dir = xml_dir_mesa

    xml_path = os.path.join(xml_dir, xml_filename)
    return xml_path

def sum_stage_wake_duration_from_csv(csv_path):
    """Sum the total 'Wake' stage duration from a CSV annotation.

    Reads the CSV file, treats the second column as the sleep-stage (Stages)
    column, and accumulates 1 second for each row whose value is ``'Wake'``
    (assuming each row represents one second).

    Args:
        csv_path (str): Path to an annotation CSV file that includes one header
            row.

    Returns:
        float: Total duration of the 'Wake' stage, in seconds.
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)

    # Extract the Stages column.
    stages_column = data[:, 1]  # Assume the second column is the Stages column.
    total_duration = 0.0

    for stage in stages_column:
        if stage == 'Wake':
            total_duration += 1.0  # Add the Wake duration (assumed 1-second units).

    return total_duration

def loadVideo(file_dir, start_frame=0, end_frame=None):
    """Load an entire video file (or a sub-range) as an array of RGB frames.

    Uses an :mod:`imageio` iterator to read frames from ``start_frame`` up to
    (but not including) ``end_frame`` and returns them as a ``uint8`` RGB
    array.

    Args:
        file_dir (str): Path to the video file (e.g. a path ending in
            ``".mp4"``).
        start_frame (int, optional): Frame index to start reading from.
            Defaults to 0.
        end_frame (int, optional): Frame index to stop reading at (exclusive).
            Reads to the end when ``None``. Defaults to ``None``.

    Returns:
        numpy.ndarray: ``uint8`` RGB frame array with shape
        ``(num_frames, H, W, 3)``.
    """
    reader = iio.imiter(file_dir)
    orig_vid = []
    for i, im in tqdm(enumerate(itertools.islice(reader, start_frame, end_frame)),
                      ascii=True,
                      desc="Load Video"):
        orig_vid.append(im)

    return np.array(orig_vid)

def loadVideoStream(reader_iter, num_frames):
    """Stream RGB frames from an active imageio iterator.

    Reads up to ``num_frames`` frames from an already-created iterator. If the
    iterator is exhausted partway through, the frames read so far are returned;
    if no frame could be read at all, ``None`` is returned. This is useful for
    processing large videos in chunks.

    Args:
        reader_iter (Iterator): Active imageio frame iterator.
        num_frames (int): Number of frames to read in this call.

    Returns:
        numpy.ndarray | None: A ``(n, H, W, 3)`` array of the frames read.
        Returns ``None`` if the stream had already ended and no frame could be
        read.
    """
    frames = []
    try:
        for _ in range(num_frames):
            frame = next(reader_iter)
            frames.append(frame)
    except StopIteration:
        # Stream ended
        if len(frames) == 0:
            return None
        # Return partial frames if any
        return np.array(frames)

    return np.array(frames)

def loadVideoStreamCV2(cap, num_frames, crop_box=None, size=None, normalize=True):
    """Stream RGB frames from an active cv2.VideoCapture object.

    Reads up to ``num_frames`` frames from ``cap``, optionally applying
    cropping, BGR-to-RGB conversion, resizing, and normalization to each frame.
    If the stream ends partway through, the frames read so far are returned; if
    no frame could be read at all, ``None`` is returned.

    Args:
        cap (cv2.VideoCapture): Active OpenCV video capture object.
        num_frames (int): Number of frames to read in this call.
        crop_box (tuple[int, int, int, int], optional): Crop region given as
            ``(y1, y2, x1, x2)``. Cropping is skipped when ``None``.
        size (tuple[int, int], optional): Target resize size given as
            ``(width, height)``. Resizing is skipped when ``None``.
        normalize (bool, optional): If ``True``, divide pixel values by 255 to
            normalize them to the ``[0, 1]`` range; if ``False``, keep
            ``uint8``. Defaults to ``True``.

    Returns:
        numpy.ndarray | None: A ``(n, H, W, 3)`` array of the processed frames.
        Returns ``None`` if the stream had already ended and no frame could be
        read.
    """
    frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Stream ended
            if len(frames) == 0:
                return None
            # Return partial frames if any
            break

        # Crop if specified
        if crop_box is not None:
            y1, y2, x1, x2 = crop_box
            frame = frame[y1:y2, x1:x2]

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if size is not None:
            w, h = size
            frame = cv2.resize(frame, (w, h))

        # Normalize if specified
        if normalize:
            frame = frame / 255.0

        frames.append(frame)

    return np.array(frames)

def FrameResize(video, size):
    """Resize every frame of a video frame array to the given size.

    Uses :func:`skimage.transform.resize` to convert all frames to the same
    size.

    Args:
        video (numpy.ndarray): Video frame array with shape
            ``(num_frames, H, W, 3)``.
        size (tuple[int, int]): Target resize size ``(width, height)``.

    Returns:
        numpy.ndarray: Resized frame array with shape
        ``(num_frames, width, height, 3)``.
    """
    w, h = size
    frames_resized = np.zeros((len(video), w, h, 3))
    for i in tqdm(range(len(video)),
                  ascii=True,
                  desc="Frame Resizing"):
        frames_resized[i] = sktransform.resize(video[i], size)

    return frames_resized

def loadFrames(frames_dir, ratio, size=None):
    """Load JPEG frames from a directory, sampled at a fixed interval.

    Selects ``ratio`` of the sorted ``.jpg`` frames in ``frames_dir``, spaced
    uniformly at intervals of ``1/ratio``, and reads them. Each frame is
    cropped to a fixed region (``[32:422, 125:515]``), converted from BGR to
    RGB, optionally resized, and then normalized to the ``[0, 1]`` range.

    Args:
        frames_dir (str): Directory containing ``.jpg`` frame images.
        ratio (float): Fraction of all frames to load (``0 < ratio <= 1``). The
            sampling interval is ``int(1 / ratio)``.
        size (tuple[int, int], optional): Target resize size
            ``(width, height)``. Resizing is skipped when ``None``.

    Returns:
        numpy.ndarray: ``[0, 1]``-normalized RGB frame array with shape
        ``(num, H, W, 3)``, where ``num = int(len(frame_names) * ratio)``.
    """
    frame_names = [f for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]

    num = int(len(frame_names)*ratio)
    interval = int(1 / ratio)
    frames = []

    for i in tqdm(range(num),
                  ascii=True,
                  desc="Load Frames"):

        frame_file = os.path.join(frames_dir, frame_names[i*interval])

        # Read frame
        frame = cv2.imread(frame_file)

        # Crop
        frame = frame[32:422, 125:515]
        #frame = frame[52:402, 145:495]
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        if size != None:
            w, h = size
            frame = cv2.resize(frame, (w, h))
        # Normalize
        frame = frame / 255.

        frames.append(frame)

    return np.array(frames)
