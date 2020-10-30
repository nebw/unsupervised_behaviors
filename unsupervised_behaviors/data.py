import decimal
import pathlib
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


def get_random_initial_frames(num_frames: int) -> Set[decimal.Decimal]:
    """Get a random sample of frame_ids at the beginning of source video files
       for the period in which detections exist in the database.

    Args:
        num_frames (int): Number of frames to sample

    Returns:
        Set[decimal.Decimal]: frame_ids of sampled frames
    """

    with bb_behavior.db.get_database_connection(application_name="unsupervised_behaviors") as con:
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


def get_image_and_mask_for_detections(
    detections: pd.DataFrame,
    video_path: pathlib.Path = None,
    video_manager: bb_behavior.io.videos.BeesbookVideoManager = None,
    image_size_px: int = 256,
    image_crop_px: int = 32,
    image_zoom_factor: float = 2 / 3,
    use_clahe: bool = True,
    clahe_kernel_size_px: int = 25,
    tag_mask_size_px: int = 18,
    bee_body_center_offset_px: int = 20,
    body_mask_length_px: int = 100,
    body_mask_width_px: int = 60,
) -> Tuple[np.array]:
    """Fetch image regions from raw BeesBook videos centered on detections of
       tagged bees. Automatically generates loss masks for ellipsoid region
       around bee based on body orientation and tags of all individuals visible
       in the image region. Automatically rotates image region according to body
       orientation.

    Args:
        detections (pd.DataFrame): Dataframe with detections.
        video_path (pathlib.Path, optional): Path to BeesBook video data. Defaults to None.
        video_manager (bb_behavior.io.videos.BeesbookVideoManager, optional): Defaults to None.
        image_size_px (int, optional): Initial image region size. Defaults to 256.
        image_crop_px (int, optional): Crop after rotation. Defaults to 32.
        image_zoom_factor (float, optional): Zoom factor after rotation. Defaults to 2/3.
        use_clahe (bool, optional): Process entire frame using CLAHE. Defaults to True.
        clahe_kernel_size_px (int, optional): Kernel size for CLAHE. Defaults to 25.
        tag_mask_size_px (int, optional): Radius of tag mask. Defaults to 18.
        bee_body_center_offset_px (int, optional): Offset from tag to body center. Defaults to 20.
        body_mask_length_px (int, optional): Length of body mask ellipsoid. Defaults to 100.
        body_mask_width_px (int, optional): Width of body mask ellipsoid. Defaults to 60.

    Returns:
         Tuple[np.array]: Extracted image regions, tag masks, and combined body
         and tag masks.
    """
    assert video_path or video_manager

    if not video_manager:
        video_manager = bb_behavior.io.videos.BeesbookVideoManager(
            str(video_path), "/home/ben/tmp/video_manager_cache/"
        )
        video_manager.cache_frames(detections.frame_id.unique())

    detections = detections[
        detections.detection_type == bb_tracking.types.DetectionType.TaggedBee.value
    ]

    images = []
    masks = []
    loss_masks = []

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
            center_y = row.y_pos - np.sin(row.orientation) * bee_body_center_offset_px
            center_x = row.x_pos - np.cos(row.orientation) * bee_body_center_offset_px
            center = np.array((center_x, center_y))

            image = bb_behavior.utils.images.get_crop_from_image(
                center, frame, width=image_size_px, clahe=False
            )
            loss_mask = np.zeros_like(frame)
            loss_mask[
                skimage.draw.ellipse(
                    center_y,
                    center_x,
                    body_mask_length_px,
                    body_mask_width_px,
                    rotation=-(row.orientation - np.pi / 2),
                    shape=frame.shape,
                )
            ] = 1
            image_mask = (
                bb_behavior.utils.images.get_crop_from_image(
                    center, tag_mask, width=image_size_px, clahe=False
                )
                == 255
            )
            loss_mask = (
                bb_behavior.utils.images.get_crop_from_image(
                    center, loss_mask, width=256, clahe=False
                )
                == 255
            )

            rotation_deg = (1 / (2 * np.pi)) * 360 * (row.orientation - np.pi / 2 + np.pi)
            image = skimage.transform.rotate(image, rotation_deg)
            image_mask = skimage.transform.rotate(image_mask, rotation_deg)
            loss_mask = skimage.transform.rotate(loss_mask, rotation_deg)

            crop = image_crop_px
            images.append(
                (
                    scipy.ndimage.zoom(image[crop:-crop, crop:-crop], image_zoom_factor, order=1)
                    * 255
                ).astype(np.uint8)
            )
            masks.append(
                scipy.ndimage.zoom(image_mask[crop:-crop, crop:-crop], image_zoom_factor, order=1)
                > 0.5
            )
            loss_masks.append(
                scipy.ndimage.zoom(
                    loss_mask[crop:-crop, crop:-crop] * image_mask[crop:-crop, crop:-crop],
                    image_zoom_factor,
                    order=1,
                )
                > 0.5
            )

    images = np.stack(images)
    masks = np.stack(masks)
    loss_masks = np.stack(loss_masks)

    return images, masks, loss_masks
