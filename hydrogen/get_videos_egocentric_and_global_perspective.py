# %%
import pathlib

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

video_root = pathlib.Path("/home/ben/ssh/aglandgraf/beesbook/2019/hd_recording/")
video_manager = bb_behavior.io.videos.BeesbookVideoManager(
    video_root=str(video_root),
    cache_path="/home/ben/tmp/beesbook_cache/",
    videos_in_subdirectories=True,
)

# %%
num_videos_total = 1000
num_frames_around = 16
min_proportion_detected = 1.0

# %%
fc_ids = data.get_random_fc_ids(num_videos_total)

# %%
def track_generator(fc_ids):
    for fc_id in fc_ids:
        yield data.get_track_detections_from_frame_container(fc_id, num_frames_around * 2 + 1)


# %%

# %%
output_path_egocentric = (
    "/home/ben/ssh/snuffles-data/public/benwild/predictive/"
    + f"videos_2019_{num_videos_total}videos_{num_frames_around}frames_egocentric.h5"
)
data.create_video_h5(output_path_egocentric, num_videos_total, num_frames_around)

data.load_and_store_videos(
    output_path_egocentric,
    track_generator(fc_ids),
    constants.Behaviors.UNKNOWN.value,
    0,
    num_videos_total,
    video_manager,
    use_clahe=True,
    n_jobs=8,
)

# %%
output_path_global = (
    "/home/ben/ssh/snuffles-data/public/benwild/predictive/"
    + f"videos_2019_{num_videos_total}videos_{num_frames_around}frames_global.h5"
)
data.create_video_h5(output_path_global, num_videos_total, num_frames_around)

data.load_and_store_videos(
    output_path_global,
    track_generator(fc_ids),
    constants.Behaviors.UNKNOWN.value,
    0,
    num_videos_total,
    video_manager,
    use_clahe=True,
    egocentric=False,
    n_jobs=8,
)

# %%
video_idx = 0
data.extract_video(output_path_egocentric, video_idx, "/home/ben/egocentric.mp4")
data.extract_video(output_path_global, video_idx, "/home/ben/global.mp4")
