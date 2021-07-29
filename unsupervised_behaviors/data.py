import datetime
import decimal
from typing import Set, Tuple

import numpy as np
import scipy
import scipy.ndimage
import skimage
import skimage.draw
import skimage.exposure
import skimage.transform
import tqdm.auto as tqdm
import pandas as pd

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_behavior.utils
import bb_behavior.utils.images
import bb_tracking
import bb_tracking.types


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

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        Extracted image regions, tag masks, and body masks.
    """

    def rotate_crop_zoom(image: np.array, rotation_deg: float) -> np.array:
        crop = image_crop_px
        image = skimage.transform.rotate(image, rotation_deg)
        image = scipy.ndimage.zoom(image[crop:-crop, crop:-crop], image_zoom_factor, order=1)
        return image

    video_manager.cache_frames(detections.frame_id.unique())

    detections = detections[
        detections.detection_type == bb_tracking.types.DetectionType.TaggedBee.value
    ]

    images = []
    tag_masks = []
    body_masks = []
    rows = []

    for _, frame_detections in tqdm.tqdm(
        detections.groupby("timestamp"), total=detections.timestamp.nunique()
    ):
        assert frame_detections.frame_id.nunique() == 1
        frame = video_manager.get_frame(frame_detections.frame_id.iloc[0])
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
            center = np.array((center_x, center_y))
            rotation_deg = (1 / (2 * np.pi)) * 360 * (row.orientation - np.pi / 2 + np.pi)

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

    images = np.stack(images)
    tag_masks = np.stack(tag_masks)
    body_masks = np.stack(body_masks)
    detections = pd.DataFrame(np.stack(rows), columns=detections.columns)

    return images, tag_masks, body_masks, detections
