# %%
import pathlib

import h5py
import matplotlib.pyplot as plt
import pandas as pd

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_behavior.utils
import bb_behavior.utils.images

import unsupervised_behaviors.constants as constants
import unsupervised_behaviors.data as data

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
num_videos_total = 10000
num_videos_dance = 1000
num_videos_following = 1000
num_videos_ventilating = 180
num_videos_behaviors = num_videos_dance + num_videos_following + num_videos_ventilating
num_frames_around = 32
min_proportion_detected = 0.9
use_clahe = True

output_path = (
    f"/srv/public/benwild/predictive/videos_2019_{num_videos_total}videos_"
    + f"{num_frames_around}frames_allbehaviors.h5"
)
data.create_video_h5(output_path, num_videos_total, num_frames_around)

# %%
trajectory_generator = data.get_random_detection_trajectory_generator(num_frames_around * 2 + 1)

data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.UNKNOWN.value,
    num_videos_behaviors,
    num_videos_total,
    video_manager,
    use_clahe=use_clahe,
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
    num_videos_dance,
    video_manager,
    use_clahe=use_clahe,
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
    num_videos_dance,
    num_videos_dance + num_videos_following,
    video_manager,
    use_clahe=use_clahe,
    n_jobs=24,
)

# %%
ventilation_df = pd.read_csv(
    "/srv/public/benwild/ventilation.csv", parse_dates=["timestamp", "from", "to"]
)

# %%
frequent_ventilation_intervals = data.get_frequent_intervals(ventilation_df)

trajectory_generator = data.get_event_detection_trajectory_generator(
    ventilation_df, frequent_ventilation_intervals.index, num_frames_around, min_proportion_detected
)

# %%
data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.VENTILATING.value,
    num_videos_dance + num_videos_following,
    num_videos_dance + num_videos_following + num_videos_ventilating,
    video_manager,
    use_clahe=True,
    n_jobs=24,
)

# %%
with h5py.File(output_path, "r") as f:
    images = f["images"]

    plt.imshow(images[0, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(images[num_videos_dance - 1, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(images[num_videos_dance, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(images[num_videos_dance + num_videos_following - 1, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(images[num_videos_dance + num_videos_following, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(
        images[num_videos_dance + num_videos_following + num_videos_ventilating - 3, 0],
        cmap=plt.cm.gray,
    )
    plt.show()

    plt.imshow(images[num_videos_behaviors, 0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(images[num_videos_total - 1, 0], cmap=plt.cm.gray)
    plt.show()

# %%
data.extract_video(output_path, -1, "/srv/public/benwild/test.mp4")
