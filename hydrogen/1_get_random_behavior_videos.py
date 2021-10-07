# %%
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
num_videos_total = 50000
num_frames_around = 32
min_proportion_detected = 0.9
use_clahe = True

output_path = (
    f"/srv/public/benwild/predictive/videos_2019_{num_videos_total}videos_"
    + f"{num_frames_around}frames_random.h5"
)
data.create_video_h5(output_path, num_videos_total, num_frames_around)

# %%
trajectory_generator = data.get_random_detection_trajectory_generator(num_frames_around * 2 + 1)

data.load_and_store_videos(
    output_path,
    trajectory_generator,
    constants.Behaviors.UNKNOWN.value,
    0,
    num_videos_total,
    video_manager,
    use_clahe=use_clahe,
    n_jobs=12,
)

# %%
with h5py.File(output_path, "r") as f:
    images = f["images"]

    plt.imshow(images[0, 0], cmap=plt.cm.gray)
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
