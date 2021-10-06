# %%
import datetime
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
wdd_truth_path = "/srv/public/dormagen/WDD2019_GroundTruth_frames_2.df.pickle"
import pandas

all_samples_df = pandas.read_pickle(wdd_truth_path)


# %%
waggle_events_df = data.get_wdd_truth_events(wdd_truth_path, "waggle")
following_events_df = data.get_wdd_truth_events(wdd_truth_path, "follower")
nothing_events_df = data.get_wdd_truth_events(wdd_truth_path, "nothing")

print(list(map(len, (waggle_events_df, following_events_df, nothing_events_df))))

# %%
num_videos_dance = len(waggle_events_df)
num_videos_following = len(following_events_df)
num_videos_nothing = len(nothing_events_df)
num_videos_total = num_videos_dance + num_videos_following + num_videos_nothing
num_frames_around = 32
min_proportion_detected = 0.1
use_clahe = True

print(num_videos_total)

output_path = (
    f"/srv/public/benwild/predictive/videos_2019_{num_videos_total}videos_"
    + f"{num_frames_around}frames_dance_follower_gt.h5"
)
data.create_video_h5(output_path, num_videos_total, num_frames_around)

# %%
behaviors = (constants.Behaviors.DANCE, constants.Behaviors.FOLLOWING, constants.Behaviors.UNKNOWN)
event_dfs = (waggle_events_df, following_events_df, nothing_events_df)
idxs = (
    (0, num_videos_dance),
    (num_videos_dance, num_videos_dance + num_videos_following),
    (num_videos_dance + num_videos_following, num_videos_total),
)

# %%
for (idx_from, idx_to), behavior, event_df in zip(idxs, behaviors, event_dfs):
    print(behavior, idx_from, idx_to)

    frequent_intervals = data.get_frequent_intervals(event_df, duration=datetime.timedelta(days=1))

    trajectory_generator = data.get_event_detection_trajectory_generator(
        event_df,
        frequent_intervals.index,
        num_frames_around,
        min_proportion_detected,
        duration=datetime.timedelta(days=1),
    )

    data.load_and_store_videos(
        output_path,
        trajectory_generator,
        behavior.value,
        idx_from,
        idx_to,
        video_manager,
        use_clahe=use_clahe,
        n_jobs=12,
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

    plt.imshow(images[num_videos_total - 1, 0], cmap=plt.cm.gray)
    plt.show()


# %%
with h5py.File(output_path, "r+") as f:
    images = f["images"]

    idxs = np.array(sorted(np.random.choice(list(range(len(images))), replace=False, size=1024)))
    sample = images[idxs]

# %%
with h5py.File(output_path, "r+") as f:
    f["mean"] = sample.mean()
    f["std"] = sample.std()

# %%
data.extract_video(output_path, -1, "/srv/public/benwild/test.mp4")

# %%
output_path
