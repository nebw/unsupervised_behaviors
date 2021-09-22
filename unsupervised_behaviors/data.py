import datetime
import decimal
import logging
import pathlib
import sys
from typing import Iterable, List, Set, Tuple

import h5py
import hdf5plugin
import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import skimage
import skimage.draw
import skimage.exposure
import skimage.io
import skimage.transform
import skvideo
import skvideo.io
import torch

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_behavior.utils
import bb_behavior.utils.images
import bb_tracking
import bb_tracking.types

import unsupervised_behaviors.constants

APPLICATION_NAME = "unsupervised_behaviors"


def get_random_initial_frames(num_frames: int) -> Set[decimal.Decimal]:
    """Get a random sample of frame_ids at the beginning of source video files
       for the period in which detections exist in the database.

    Parameters
    ----------
    num_frames : int
        Number of frames to sample

    Returns
    -------
    Set[decimal.Decimal]
        frame_ids of sampled frames
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        frame_df = pd.read_sql(
            f"""
                SELECT frame_id FROM {bb_behavior.db.get_frame_metadata_tablename()}
                WHERE index = 0
                AND datetime >= (
                    SELECT MIN(timestamp)
                    FROM {bb_behavior.db.get_detections_tablename()}
                )
                AND datetime <= (
                    SELECT MAX(timestamp)
                    FROM {bb_behavior.db.get_detections_tablename()}
                )
                ORDER BY RANDOM()
                LIMIT {num_frames}
            """,
            coerce_float=False,
            con=con,
        )

    return set(frame_df.frame_id)


def get_inital_frame_pair_dataframe(date: datetime.date, cam_id: int) -> pd.DataFrame:
    """Get dataframe with first two frames of all videos for given date.

    Parameters
    ----------
    date : datetime.date
        Date for which to fetch frames.
    cam_id : int
        Only fetch frames from this camera.

    Returns
    -------
    pd.DataFrame
        [description]
        Dataframe with entries from frame metadata table for requested date.
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        frame_df = pd.read_sql(
            f"""
                SELECT * FROM {bb_behavior.db.get_frame_metadata_tablename()}
                WHERE index < 2
                AND datetime >= %s
                AND datetime < %s
                AND cam_id = {cam_id}
            """,
            con,
            parse_dates=True,
            coerce_float=False,
            params=(date, date + datetime.timedelta(hours=24)),
        )

    return frame_df


def get_random_fc_ids(num_fcs: int) -> Set[decimal.Decimal]:
    """Get a random sample of framecontainer ids (corresponding to videos) for the period in which
    detections exist in the database.

    Parameters
    ----------
    num_fcs : int
        Number of framecontainer ids to sample

    Returns
    -------
    Set[decimal.Decimal]
        ids of sampled videos
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        fc_df = pd.read_sql(
            f"""
                SELECT fc_id FROM (
                    SELECT
                        fc_id,
                        MIN(datetime) AS MIN_TIMESTAMP,
                        MAX(datetime) AS MAX_TIMESTAMP
                    FROM {bb_behavior.db.get_frame_metadata_tablename()}
                    GROUP BY fc_id
                ) subquery
                WHERE
                    MIN_TIMESTAMP >= (
                        SELECT MIN(timestamp)
                        FROM {bb_behavior.db.get_detections_tablename()}
                    )
                    AND MAX_TIMESTAMP <= (
                        SELECT MAX(timestamp)
                        FROM {bb_behavior.db.get_detections_tablename()}
                    )
                ORDER BY RANDOM()
                LIMIT {num_fcs}
            """,
            coerce_float=False,
            con=con,
        )

    return set(fc_df.fc_id)


def get_full_track_detections(fc_id: decimal.Decimal, num_frames: int) -> pd.DataFrame:
    """Get all detections from bees that have detections on all first `num_frames` frames of the
    video with id `fc_id`.

    Parameters
    ----------
    fc_id : decimal.Decimal
        Framecontainer ID to get detections from
    num_frames : int
        Number of initial frames in Framecontainer to consider

    Returns
    -------
    pd.DataFrame
        Detections from the given Framecontainer
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        # return track_ids of all tracks in this frame container with a detection in each of the
        # first `num_frames` frames
        track_df = pd.read_sql(
            f"""
            SELECT track_id FROM (
                SELECT DISTINCT(track_id), COUNT(track_id) AS NUM_DETECTIONS FROM (
                    SELECT frame_id
                    FROM {bb_behavior.db.get_frame_metadata_tablename()}
                    WHERE fc_id = {fc_id}
                    ORDER BY index
                    LIMIT {num_frames}
                ) frame_query
                JOIN {bb_behavior.db.get_detections_tablename()} AS D
                ON frame_query.frame_id = D.frame_id
                GROUP BY track_id
            ) track_query
            WHERE NUM_DETECTIONS = {num_frames}
            """,
            con=con,
            coerce_float=False,
        )

        # frame_ids of the same first `num_frames` frames
        frame_df = pd.read_sql(
            f"""
            SELECT frame_id
            FROM {bb_behavior.db.get_frame_metadata_tablename()}
            WHERE fc_id = {fc_id}
            ORDER BY index
            LIMIT {num_frames}
            """,
            con=con,
            coerce_float=False,
        )

        # all detections from the previously selected frames and tracks
        detection_df = pd.read_sql(
            f"""
            SELECT *
            FROM {bb_behavior.db.get_detections_tablename()}
            WHERE
                track_id IN {tuple(map(int, track_df.track_id))} AND
                frame_id IN {tuple(map(int, frame_df.frame_id))}
            """,
            con=con,
            coerce_float=False,
        )

        return detection_df


def get_image_and_mask_for_detections(
    detections: pd.DataFrame,
    video_manager: bb_behavior.io.videos.BeesbookVideoManager,
    image_size_px: int = 256,
    image_crop_px: int = 32,
    image_zoom_factor: float = 2 / 3,
    use_clahe: bool = True,
    clahe_kernel_size_px: int = 25,
    tag_mask_size_px: int = 18,
    body_center_offset_px: int = 20,
    body_mask_length_px: int = 100,
    body_mask_width_px: int = 60,
    egocentric: bool = True,
    n_jobs: int = -1,
) -> Tuple[np.array, np.array, np.array]:
    """Fetch image regions from raw BeesBook videos centered on detections of
       tagged bees. Automatically generates loss masks for ellipsoid region
       around bee based on body orientation and tags of all individuals visible
       in the image region. Automatically rotates image region according to body
       orientation.

    Parameters
    ----------
    detections : pd.DataFrame
        Dataframe with detections.
    video_manager : bb_behavior.io.videos.BeesbookVideoManager
        Video cache.
    image_size_px : int, optional
        Initial image region size. Defaults to 256.
    image_crop_px : int, optional
        Crop after rotation. Defaults to 32.
    image_zoom_factor : float, optional
        Zoom factor after rotation. Defaults to 2/3.
    use_clahe : bool, optional
        Process entire frame using CLAHE. Defaults to True.
    clahe_kernel_size_px : int, optional
        Kernel size for CLAHE. Defaults to 25.
    tag_mask_size_px : int, optional
        Radius of tag mask. Defaults to 18.
    body_center_offset_px : int, optional
        Offset from tag to body center. Defaults to 20.
    body_mask_length_px : int, optional
        Length of body mask ellipsoid. Defaults to 100.
    body_mask_width_px : int, optional
        Width of body mask ellipsoid. Defaults to 60.
    egocentric : bool, optional
        Rotate frame using bee orientation such that it is egocentric.
    n_jobs : int, optional
        Number of parallel jobs for processing.

    Returns
    -------
    Tuple[np.array, np.array, np.array, pd.DataFrame]
        Extracted image regions, tag masks, body masks, and a dataframe with detections in the
        same order as the images and masks.
    """

    def rotate_crop_zoom(image: np.array, rotation_deg: float) -> np.array:
        crop = image_crop_px
        image = skimage.transform.rotate(image, rotation_deg)
        image = scipy.ndimage.zoom(image[crop:-crop, crop:-crop], image_zoom_factor, order=1)
        return image

    def extract_images(
        frame_detections: pd.DataFrame, frame_path: str
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        images = []
        tag_masks = []
        body_masks = []
        rows = []

        frame = skimage.io.imread(frame_path, as_gray=True, plugin="matplotlib")

        if use_clahe:
            frame = skimage.exposure.equalize_adapthist(frame, clahe_kernel_size_px)

        tag_mask = np.ones_like(frame)
        for _, row in frame_detections.iterrows():
            tag_mask[
                skimage.draw.disk((row.y_pos, row.x_pos), tag_mask_size_px, shape=frame.shape)
            ] = 0

        for _, row in frame_detections.iterrows():
            center_y = row.y_pos - np.sin(row.orientation) * body_center_offset_px
            center_x = row.x_pos - np.cos(row.orientation) * body_center_offset_px

            # assert center point always within frame (even after trajectory extrapolation)
            center_x = max(0, center_x)
            center_x = min(frame.shape[1] - 1, center_x)
            center_y = max(0, center_y)
            center_y = min(frame.shape[0] - 1, center_y)

            center = np.array((center_x, center_y))

            if egocentric:
                rotation_deg = (1 / (2 * np.pi)) * 360 * (row.orientation - np.pi / 2 + np.pi)
            else:
                rotation_deg = 0

            image = bb_behavior.utils.images.get_crop_from_image(
                center, frame, width=image_size_px, clahe=False
            )

            image = (rotate_crop_zoom(image, rotation_deg) * 255).astype(np.uint8)
            images.append(image)

            task_mask = (
                bb_behavior.utils.images.get_crop_from_image(
                    center, tag_mask, width=image_size_px, clahe=False
                )
                == 255
            )
            task_mask = rotate_crop_zoom(task_mask, rotation_deg) > 0.5
            tag_masks.append(task_mask)

            body_mask = np.zeros_like(frame)
            body_coords = skimage.draw.ellipse(
                center_y,
                center_x,
                body_mask_length_px,
                body_mask_width_px,
                rotation=-(row.orientation - np.pi / 2),
                shape=frame.shape,
            )
            body_mask[body_coords] = 1
            body_mask = (
                bb_behavior.utils.images.get_crop_from_image(
                    center, body_mask, width=image_size_px, clahe=False
                )
                == 255
            )
            body_mask = rotate_crop_zoom(body_mask, rotation_deg) > 0.5
            body_masks.append(body_mask)

            rows.append(row.values)

        return images, tag_masks, body_masks, rows

    video_manager.cache_frames(detections.frame_id.unique())

    detections = detections[
        detections.detection_type == bb_tracking.types.DetectionType.TaggedBee.value
    ]

    images = []
    tag_masks = []
    body_masks = []
    rows = []

    detections_by_frame = detections.groupby("timestamp")

    # preload file paths because video_manager can't be used in parallel.
    frame_paths = []
    for _, frame_detections in detections_by_frame:
        assert frame_detections.frame_id.nunique() == 1
        frame_paths.append(video_manager.get_frame_id_path(frame_detections.frame_id.iloc[0]))

    parallel = joblib.Parallel(prefer="processes", n_jobs=n_jobs)(
        joblib.delayed(extract_images)(frame_detections, frame_path)
        for (_, frame_detections), frame_path in zip(detections_by_frame, frame_paths)
    )

    for results in parallel:
        images += results[0]
        tag_masks += results[1]
        body_masks += results[2]
        rows += results[3]

    images = np.stack(images)
    tag_masks = np.stack(tag_masks)
    body_masks = np.stack(body_masks)
    detections = pd.DataFrame(np.stack(rows), columns=detections.columns)

    return images, tag_masks, body_masks, detections


def get_frequent_intervals(
    event_df: pd.DataFrame, duration: datetime.timedelta = datetime.timedelta(minutes=1)
) -> pd.DataFrame:
    """Get intervals and camera IDs sorted by counts of events, such that intervals with many
    events can be processed first.

    Parameters
    ----------
    event_df: pd.DataFrame
        Event DataFrame, containing e.g. dance or following events with timestamps.
    duration: datetime.timedelta
        Interval length.

    Returns
    -------
    pd.DataFrame
        Intervals sorted by event counts. DataFrame with MultiIndex [cam_id, timestamp]
    """
    frequent_intervals = (
        event_df.groupby([pd.Grouper(key="cam_id"), pd.Grouper(key="from", freq=duration)])
        .bee_id.count()
        .sort_values()[::-1]
    )

    return frequent_intervals


def convert_bb_behavior_trajectories(
    results: Iterable[Tuple[np.array, np.array, int]]
) -> pd.DataFrame:
    """Convert trajectories as returned by `get_event_detection_trajectory_generator` into a
    DataFrame that can be used by `get_image_and_mask_for_detections`. This is necessary because
    `get_image_and_mask_for_detections` iterates over frames instead of trajectories to avoid
    unnecessarily loading the same image twice.

    Parameters
    ----------
    results: Iterable[Tuple[np.array, np.array, int]]
        Trajectory generator as returned by `get_event_detection_trajectory_generator`.

    Returns
    -------
    pd.DataFrame
        DataFrame with all detections from the trajectories, with column `video_idx` as a sequential
        index of the trajectory.
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        detections = []
        for idx, (frame_ids, traj, bee_id) in enumerate(results):
            frames = pd.read_sql(
                f"""
                SELECT frame_id, timestamp, cam_id
                FROM {bb_behavior.db.get_frame_metadata_tablename()}
                WHERE
                    frame_id IN {tuple(frame_ids)}
                """,
                con=con,
                coerce_float=False,
            )
            trajectory_df = pd.DataFrame(traj, columns=("x_pos", "y_pos", "orientation"))
            trajectory_df["frame_id"] = frame_ids
            trajectory_df["bee_id"] = bee_id
            trajectory_df["detection_type"] = bb_tracking.types.DetectionType.TaggedBee.value
            trajectory_df["video_idx"] = idx

            trajectory_df = trajectory_df.merge(frames, on="frame_id")

            detections.append(trajectory_df)

        detection_df = pd.concat(detections)

        return detection_df


def get_event_detection_trajectory_generator(
    events_df: pd.DataFrame,
    timestamps: Iterable[Tuple[int, np.datetime64]],
    num_frames_around: int,
    min_proportion_detected: float,
    duration: datetime.timedelta = datetime.timedelta(minutes=1),
) -> Iterable[pd.DataFrame]:
    """Given intervals (from get_frequent_intervals), yield consistent trajectories (position,
    orientation, frame_id) of event individuals (e.g. dancing bees). Optionaly filter by proportion
    of detections, e.g. only return trajectory if all frames have a detection. Interpolate missing
    detections otherwise.

    Parameters
    ----------
    event_df: pd.DataFrame
        Event DataFrame, containing e.g. dance or following events with timestamps.
    timestamps: Iterable[Tuple[int, np.datetime64]]
        Iterator over camera ID and timestamp, e.g. obtained by `get_frequent_intervals`.
    duration: datetime.timedelta = datetime.timedelta(minutes=1)
        Extract all events within t + duration of each timestamp.
    num_frames_around: int
        Number of frames around each event to extract, e.g. for 16 the 16 preceding and following
        frames will be extracted, for a total of 33 frames.
    min_proportion_detected: float
        Optionaly only return trajectories with at least the given proportion of frames with valid
        detections. If < 1, interpolate missing detections.

    Returns
    -------
    Iterable[pd.DataFrame]
        Generator of DataFrames with positions, orientations, frame and track IDs, and detection
        indices.
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        for cam_id, timestamp in timestamps:
            results = []

            events_subset_df = events_df[
                (events_df["from"] >= timestamp)
                & (events_df["to"] <= (timestamp + duration))
                & (events_df["cam_id"] == cam_id)
            ]

            for event in events_subset_df.iterrows():
                event = event[1]

                # return track_ids of all tracks in this frame container with a detection in each of
                # the first `num_frames` frames
                detection_df = pd.read_sql(
                    f"""
                    SELECT frame_id
                    FROM {bb_behavior.db.get_detections_tablename()}
                    WHERE
                        bee_id = {event.bee_id} AND
                        cam_id = {event.cam_id} AND
                        timestamp >= %s AND
                        timestamp <= %s
                    ORDER BY timestamp
                    """,
                    params=[event["from"], event["to"]],
                    con=con,
                    coerce_float=False,
                )

                if not len(detection_df):
                    continue

                frame_id = detection_df.frame_id.iloc[len(detection_df) // 2]

                frame_ids = bb_behavior.db.sampling.get_neighbour_frames(
                    frame_id=frame_id, n_frames=num_frames_around
                )
                frame_ids = [t[1] for t in frame_ids]

                traj, mask = bb_behavior.db.get_interpolated_trajectory(
                    int(event.bee_id), frames=frame_ids
                )

                if np.mean(mask) >= min_proportion_detected:
                    results.append((frame_ids, traj, event.bee_id))

            try:
                yield convert_bb_behavior_trajectories(results)
            except Exception as err:
                logging.getLogger().warning(f"Error loading event: {event} - {err}")


def create_video_h5(h5_path: pathlib.Path, num_videos: int, num_frames_around: int):
    """Create output h5 file for videos, masks, and labels. Uses compression.

    Parameters
    ----------
    h5_path: pathlib.Path
        Output path.
    num_videos: int
        Total number of videos to store.
    num_frames_around:
        Number of frames around each frame, used to determine number of frames per video.
    """
    with h5py.File(h5_path, "w", libver="latest") as f:
        shape = (num_videos, num_frames_around * 2 + 1, 128, 128)
        chunks = (1, num_frames_around * 2 + 1, 128, 128)

        f.create_dataset("images", shape, chunks=chunks, dtype="u8", **hdf5plugin.Zstd())
        f.create_dataset("tag_masks", shape, chunks=chunks, dtype=bool, **hdf5plugin.Zstd())
        f.create_dataset("loss_masks", shape, chunks=chunks, dtype=bool, **hdf5plugin.Zstd())
        f.create_dataset("frame_ids", (num_videos, num_frames_around * 2 + 1), dtype=np.uint)
        f.create_dataset("x_pos", (num_videos, num_frames_around * 2 + 1), dtype=np.float32)
        f.create_dataset("y_pos", (num_videos, num_frames_around * 2 + 1), dtype=np.float32)
        f.create_dataset("labels", (num_videos,), dtype=int)


def get_track_detections_from_frame_container(fc_id: decimal.Decimal, num_frames: int):
    """Retrieve a dataframe with detections from consistent tracks at the beginning of the
    framecontainer with the given ID. I.e., the DataFrame will contain detections of all individuals
    in the initial frames of the video that have detection in each frame.

    Parameters
    ----------
    fc_id: decimal.Decimal
        Framecontainer ID.
    num_frames: int
        Number of initial frames to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame with positions, orientations, frame and track IDs.
    """
    with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
        # return track_ids of all tracks in this frame container with a detection in each of the
        # first `num_frames` frames
        track_df = pd.read_sql(
            f"""
            SELECT track_id FROM (
                SELECT DISTINCT(track_id), COUNT(track_id) AS NUM_DETECTIONS FROM (
                    SELECT frame_id
                    FROM {bb_behavior.db.get_frame_metadata_tablename()}
                    WHERE fc_id = {fc_id}
                    ORDER BY index
                    LIMIT {num_frames}
                ) frame_query
                JOIN {bb_behavior.db.get_detections_tablename()} AS D
                ON frame_query.frame_id = D.frame_id
                    AND D.detection_type = {bb_tracking.types.DetectionType.TaggedBee.value}
                GROUP BY track_id
            ) track_query
            WHERE NUM_DETECTIONS = {num_frames}
            """,
            con=con,
            coerce_float=False,
        )

        # frame_ids of the same first `num_frames` frames
        frame_df = pd.read_sql(
            f"""
            SELECT frame_id
            FROM {bb_behavior.db.get_frame_metadata_tablename()}
            WHERE fc_id = {fc_id}
            ORDER BY index
            LIMIT {num_frames}
            """,
            con=con,
            coerce_float=False,
        )

        # all detections from the previously selected frames and tracks, i.e. all detections from bees
        # that have detections on all first `num_frames` frames of the video with id `fc_id`
        detection_df = pd.read_sql(
            f"""
            SELECT *
            FROM {bb_behavior.db.get_detections_tablename()}
            WHERE
                track_id IN {tuple(map(int, track_df.track_id))} AND
                frame_id IN {tuple(map(int, frame_df.frame_id))}
            """,
            con=con,
            coerce_float=False,
        )

    return detection_df


def get_random_detection_trajectory_generator(
    num_frames: int, num_cache_fc_ids: int = 100
) -> Iterable[pd.DataFrame]:
    """Yield consistent trajectories (position, orientation, frame_id) of individuals. Select
    trajectories from initial frames from frame containers such that as few frames as possible need
    to be decoded and processed.

    Parameters
    ----------
    num_frames: int
        Number of frames per video
    num_cache_fc_ids: int
        Size of framecontainer ID cache. Selecting a random ID from the database is rather slow,
        therefore we precache a number of IDs at once.

    Returns
    -------
    Iterable[pd.DataFrame]
        Generator of DataFrames with positions, orientations, frame and track IDs.
    """
    fc_ids = set()
    while True:
        if not len(fc_ids):
            fc_ids.update(unsupervised_behaviors.data.get_random_fc_ids(num_cache_fc_ids))

        yield get_track_detections_from_frame_container(fc_ids.pop(), num_frames)


def load_and_store_videos(
    h5_path: pathlib.Path,
    trajectory_generator: Iterable[pd.DataFrame],
    label: unsupervised_behaviors.constants.Behaviors,
    from_idx: int,
    to_idx: int,
    video_manager: bb_behavior.io.videos.BeesbookVideoManager,
    use_clahe: bool = True,
    clear_video_cache: bool = True,
    egocentric: bool = True,
    verbose: bool = True,
    n_jobs: int = -1,
) -> List[pd.DataFrame]:
    """For each trajectory, load images and masks and store them in the given h5 file.

    Parameters
    ----------
    h5_path: pathlib.Path
        Output file path.
    trajectory_generator: Iterable[pd.DataFrame]
        Trajectory generator as returned by `get_*_detection_trajectory_generator`.
    label: unsupervised_behaviors.constants.Behaviors
        Label of events returned by `trajectory_generator`.
    from_idx: int
        Store videos in h5 file starting at index `from_idx`.
    to_idx: int
        Store videos until index `from_idx`, i.e. extract and store `to_idx - from_idx` trajectories
        in total.
    video_manager: bb_behavior.io.videos.BeesbookVideoManager
        Video cache.
    use_clahe: bool, optional
        Process entire frame using CLAHE. Defaults to True.
    egocentric: bool, optional
        Rotate each frame such that it is egocentric.
    clear_video_cache: bool, optional
        Clear video cache at end of processing.
    verbose: bool, optional
        Print total number of loaded videos after each trajectory interval.
    n_jobs : int, optional
        Number of parallel jobs for processing.
    """
    with h5py.File(h5_path, "r+") as f:
        images_dset = f["images"]
        tag_masks_dset = f["tag_masks"]
        loss_masks_dset = f["loss_masks"]
        labels_dset = f["labels"]
        frame_ids_dset = f["frame_ids"]
        x_pos_dset = f["x_pos"]
        y_pos_dset = f["y_pos"]

        idx = from_idx
        for detection_df in trajectory_generator:
            all_frame_ids = detection_df.frame_id.unique()
            video_manager.cache_frames(all_frame_ids)

            (images, masks, loss_masks, detection_df,) = get_image_and_mask_for_detections(
                detection_df,
                video_manager,
                use_clahe=use_clahe,
                n_jobs=n_jobs,
                egocentric=egocentric,
            )

            # if generator is event generator: video_idx, if random generator: track_id
            group_column = "video_idx" if "video_idx" in detection_df.columns else "track_id"

            for _, group in detection_df.groupby(group_column):
                idxs = group.sort_values("timestamp").index.values

                images_dset[idx, :] = images[idxs]
                tag_masks_dset[idx, :] = masks[idxs]
                loss_masks_dset[idx, :] = loss_masks[idxs]
                labels_dset[idx] = label
                frame_ids_dset[idx, :] = group.frame_id[:]
                x_pos_dset[idx, :] = group.x_pos[:]
                y_pos_dset[idx, :] = group.y_pos[:]

                idx += 1

                if idx == to_idx:
                    if clear_video_cache:
                        video_manager.clear_video_cache()

                    if verbose:
                        sys.stdout.write(f"\rLoaded {idx}/{to_idx} videos.")

                    return

            if verbose:
                sys.stdout.write(f"\rLoaded {idx}/{to_idx} videos.")


def extract_video(
    h5_path: pathlib.Path, video_idx: int, output_path: pathlib.Path, with_mask: bool = False
):
    """Extract a single video from the h5 file and store it in a compressed video.

    Parameters
    ----------
    h5_path: pathlib.Path
        Video h5 file path.
    video_idx: int
        Sequential index of video to extract.
    output_path: pathlib.Path
        Output video path.
    with_mask: bool
        Mask out tags and background.
    """
    with h5py.File(h5_path, "r") as f:

        video = f["images"][video_idx]

        if with_mask:
            mask = f["tag_masks"][video_idx] * f["loss_masks"][video_idx]
            video *= mask

        outputdict = {"-c:v": "libx264", "-crf": "0", "-preset": "veryslow", "-filter:v": "fps=6"}

        with skvideo.io.FFmpegWriter(output_path, outputdict=outputdict) as writer:
            for frame in video:
                writer.writeFrame(frame[:, :, None].repeat(3, axis=-1))


class MaskedFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        transform=None,
        target_transform=None,
        ellipse_masked=True,
        tag_masked=True,
        horizontal_crop=None,
    ):
        self.file = h5py.File(path, "r")

        self.images = self.file["images"]
        self.tag_masks = self.file["tag_masks"]
        self.loss_masks = self.file["loss_masks"]
        self.labels = self.file["labels"]
        self.ellipse_masked = ellipse_masked
        self.tag_masked = tag_masked
        self.horizontal_crop = horizontal_crop

        self.transform = transform
        self.target_transform = target_transform

        self.is_video_dataset = self.images.ndim == 4

    def __len__(self):
        if self.is_video_dataset:
            return self.images.shape[0] * self.images.shape[1]
        else:
            return len(self.images)

    def apply_masks(self, image, idx, frame_idx=None):
        def _apply_mask(image, mask_dset):
            if self.is_video_dataset:
                mask = mask_dset[idx, frame_idx]
            else:
                mask = mask_dset[idx]
            return image * mask

        if self.tag_masked:
            image = _apply_mask(image, self.tag_masks)

        if self.ellipse_masked:
            image = _apply_mask(image, self.loss_masks)

        return image

    def __getitem__(self, idx):
        if self.is_video_dataset:
            idx, frame_idx = divmod(idx, self.images.shape[1])
            # frame_idx = self.images.shape[1] // 2
            # frame_idx = np.random.randint(0, self.images.shape[1])
            image = self.images[idx, frame_idx]
        else:
            image = self.images[idx]
            frame_idx = None

        image = self.apply_masks(image, idx, frame_idx)
        label = self.labels[idx]

        if self.horizontal_crop is not None:
            image = image[:, self.horizontal_crop : -self.horizontal_crop]

        image = np.expand_dims(image, -1).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
