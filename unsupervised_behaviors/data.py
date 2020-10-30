import decimal
from typing import Set

import pandas as pd

import bb_behavior
import bb_behavior.db


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
