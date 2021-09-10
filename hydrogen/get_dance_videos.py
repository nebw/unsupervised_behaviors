# %%
import pathlib

import pandas as pd

import unsupervised_behaviors.constants as constants
import unsupervised_behaviors.data as data

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_behavior.utils
import bb_behavior.utils.images

APPLICATION_NAME = "unsupervised_behaviors"

# %%
bb_behavior.db.set_season_berlin_2019()

video_root = pathlib.Path("/srv/public/benwild/trove/beesbook/2019/hd_recording/")
video_manager = bb_behavior.io.videos.BeesbookVideoManager(
    video_root=str(video_root),
    cache_path="/srv/data/benwild/cache/",
    videos_in_subdirectories=True,
)

# %%
num_videos_total = 20000
num_videos_per_behavior = 1000
num_frames_around = 32
min_proportion_detected = 1.0

output_path = f"/srv/public/benwild/predictive/videos_2019_{num_videos_total}videos_{num_frames_around}frames.h5"
data.create_video_h5(output_path, num_videos_total, num_frames_around)

# %%
trajectory_generator = data.get_random_detection_trajectory_generator(num_frames_around * 2 + 1)

data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.UNKNOWN.value,
    num_videos_per_behavior * 2,
    num_videos_total,
    video_manager,
    use_clahe=True,
    n_jobs=24,
)

# %%
dances_df = pd.read_pickle("/srv/public/dormagen/all_dances.pickle")

frequent_dance_intervals = data.get_frequent_intervals(dances_df)

trajectory_generator = data.get_event_detection_trajectory_generator(
    dances_df, frequent_dance_intervals.index, num_frames_around, min_proportion_detected
)

# %%
data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.DANCE.value,
    0,
    num_videos_per_behavior,
    video_manager,
    use_clahe=True,
    n_jobs=24,
)

# %%
following_df = pd.read_pickle("/srv/public/dormagen/all_followers.pickle")
following_df["from"] = following_df["timestamp_from"]
following_df["to"] = following_df["timestamp_to"]
following_df["bee_id"] = following_df["follower_id"]

# %%
frequent_following_intervals = data.get_frequent_intervals(following_df)

trajectory_generator = data.get_event_detection_trajectory_generator(
    following_df, frequent_following_intervals.index, num_frames_around, min_proportion_detected
)

data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.FOLLOWING.value,
    num_videos_per_behavior,
    num_videos_per_behavior * 2,
    video_manager,
    use_clahe=True,
    n_jobs=24,
)

# %%
data.extract_video(output_path, num_videos_per_behavior * 2, "/srv/public/benwild/test.mp4")
