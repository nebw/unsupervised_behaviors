import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import scipy
import scipy.ndimage
import skimage
import skimage.draw
import skimage.morphology
from typing import Tuple

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_tracking
import bb_tracking.types
import pipeline

from unsupervised_behaviors import utils


def get_saliency_pipeline():
    saliency_pipeline = pipeline.Pipeline(
        [pipeline.objects.Image],
        [pipeline.objects.SaliencyImages],
        **pipeline.pipeline.get_auto_config(),
    )
    localizer = [s for s in saliency_pipeline.stages if isinstance(s, pipeline.stages.Localizer)][0]
    class_labels = localizer.class_labels
    return saliency_pipeline, class_labels


def get_background_histogram(
    frame_df: pd.DataFrame,
    video_manager: bb_behavior.io.videos.BeesbookVideoManager,
    diff_threshold: float = 1e-3,
    saliency_threshold: float = 0.1,
    dilation_kernel_size: int = 6,
    body_center_offset_px: int = 20,
    body_mask_length_px: int = 100,
    body_mask_width_px: int = 60,
) -> Tuple[np.array, np.array]:
    """Extract "smart" background image histogram from frames in frame_df by
       selecting pixels with low movement, and no detections of bees for each frame.

    Args:
        frame_df (pd.DataFrame):
            DataFrame obtained via data.get_inital_frame_pair_dataframe.
        video_manager (bb_behavior.io.videos.BeesbookVideoManager):
            Video cache.
        diff_threshold (float):
            Maximum euclidean distance in image differential. Defaults to 1e-3.
        saliency_threshold (float, optional):
            Maximum localizer saliency. Defaults to 0.1.
        dilation_kernel_size (int, optional):
            Localizer saliency kernel size. Defaults to 6.
        body_center_offset_px (int, optional):
            Offset from tag to body center. Defaults to 20.
        body_mask_length_px (int, optional):
            Length of body mask ellipsoid. Defaults to 100.
        body_mask_width_px (int, optional):
            Width of body mask ellipsoid. Defaults to 60.

    Returns:
        Tuple[np.array, np.array]: Sum and counts of background pixels.
    """

    video_manager.cache_frames(frame_df.frame_id.unique())
    saliency_pipeline, class_labels = get_saliency_pipeline()

    counts = np.zeros((3000, 4000, 256), dtype=np.float32)
    sums = np.zeros((3000, 4000), dtype=np.float32)
    dilation_kernel = skimage.morphology.disk(dilation_kernel_size)

    for _, video_frames in tqdm.tqdm(frame_df.groupby("fc_id"), total=frame_df.fc_id.nunique()):
        last_frame = video_manager.get_frame(video_frames.frame_id.iloc[0])
        frame = video_manager.get_frame(video_frames.frame_id.iloc[1])

        diff = (frame - last_frame) ** 2
        selection = diff < diff_threshold

        detections = bb_behavior.db.get_detections_dataframe_for_frames(
            [video_frames.frame_id.iloc[1]],
            use_hive_coordinates=False,
            additional_columns=["detection_type"],
            confidence_threshold=0,
        )

        # select pixels without detections of tagged bees
        detections = detections[
            detections.detection_type == bb_tracking.types.DetectionType.TaggedBee.value
        ]
        for _, row in detections.iterrows():
            center_y = row.y_pos - np.sin(row.orientation) * body_center_offset_px
            center_x = row.x_pos - np.cos(row.orientation) * body_center_offset_px
            body_mask = skimage.draw.ellipse(
                center_y,
                center_x,
                body_mask_length_px,
                body_mask_width_px,
                rotation=-(row.orientation - np.pi / 2),
                shape=frame.shape,
            )
            selection[body_mask] = False

        # select pixels with low localizer saliency for untagged bees
        frame_u8 = (frame * 255).astype(np.uint8)
        saliency = saliency_pipeline([frame_u8])[pipeline.objects.SaliencyImages]
        saliency = saliency[:, :, [label != "MarkedBee" for label in class_labels]].max(axis=-1)
        saliency = skimage.morphology.dilation(saliency, dilation_kernel)

        # FIXME; Don't hardcode scaling factor
        saliencies_scaled = scipy.ndimage.zoom(saliency, (3000 / 376, 4000 / 501))
        selection *= saliencies_scaled < saliency_threshold

        sums[selection] += frame[selection]
        utils.increase_histogram(counts, frame_u8, selection)

    return sums, counts
