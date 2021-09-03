# %%
import pathlib

import pandas as pd

import unsupervised_behaviors
import unsupervised_behaviors.data as data

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_behavior.utils
import bb_behavior.utils.images
import bb_tracking
import bb_tracking.types

APPLICATION_NAME = "unsupervised_behaviors"

# %%
bb_behavior.db.set_season_berlin_2019()
bb_behavior.db.get_framecontainer_metadata_tablename()

# %%
fc_ids = unsupervised_behaviors.data.get_random_fc_ids(10)

# %%
fc_ids

# %%
with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
    frame_df = pd.read_sql(
        f"""
            SELECT column_name
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{bb_behavior.db.get_detections_tablename()}';
        """,
        con=con,
    )

frame_df

# %%


# %%
num_frames = 33
use_clahe = True
n_jobs = 24

for fc_id in fc_ids:
    detection_df = get_detections_from_frame_container(fc_id, num_frames)
    break

# %%
video_root = pathlib.Path("/srv/public/benwild/trove/2019/hd_recording/")
video_manager = bb_behavior.io.videos.BeesbookVideoManager(
    video_root=str(video_root),
    cache_path="/srv/data/benwild/cache/",
    videos_in_subdirectories=True,
)

# %%
(images, masks, loss_masks, detection_df,) = data.get_image_and_mask_for_detections(
    detection_df, video_manager, use_clahe=use_clahe, n_jobs=n_jobs
)

# %%
len(images)

detection_df.track_id.nunique()
