# %%
import datetime
import json
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.draw
from fastprogress.fastprogress import force_console_behavior
from shared.plotting import setup_matplotlib

import unsupervised_behaviors
import unsupervised_behaviors.data
from unsupervised_behaviors.constants import Behaviors

import bb_behavior
import bb_behavior.db
import bb_behavior.io
import bb_tracking.types

APPLICATION_NAME = "unsupervised_behaviors"

bb_behavior.db.set_season_berlin_2019()

master_bar, progress_bar = force_console_behavior()
setup_matplotlib()

# %%
path = "/home/ben/Downloads/label.json"
data = json.load(open(path))

# %%
rows = []
for frame in data:
    if frame["External ID"].startswith("Cam"):
        video_name = frame["Dataset Name"]
        frame_number = int(frame["External ID"].split("_")[-1].split(".jpg")[0])

        if "objects" not in frame["Label"]:
            continue

        for datapoint in frame["Label"]["objects"]:
            x_start = datapoint["line"][0]["x"]
            y_start = datapoint["line"][0]["y"]
            x_end = datapoint["line"][1]["x"]
            y_end = datapoint["line"][1]["y"]

            rows.append(
                dict(
                    video_name=video_name,
                    frame_number=frame_number,
                    x_start=x_start,
                    y_start=y_start,
                    x_end=x_end,
                    y_end=y_end,
                )
            )

df = pd.DataFrame(rows)


# %%


def parse_video_name(video_name):
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    video_name = video_name.replace("mp4", "avi")
    cam_prefix, part1, part2 = video_name.split("T")
    part1 = part1.replace("_", ":")
    part2 = part2.replace("_", ":")
    video_name = "T".join((cam_prefix, part1, part2))
    return video_name


df.video_name = df.video_name.apply(parse_video_name)

# %%
with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
    fc_df = pd.read_sql(
        f"""
            SELECT fc_id, video_name
            FROM {bb_behavior.db.get_framecontainer_metadata_tablename()}
            WHERE video_name IN {tuple(set(df.video_name))}
        """,
        coerce_float=False,
        con=con,
    )

df = df.merge(fc_df, on="video_name")

# %%
with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
    frame_df = pd.read_sql(
        f"""
            SELECT frame_id, fc_id, index as frame_number
            FROM {bb_behavior.db.get_frame_metadata_tablename()}
            WHERE fc_id IN {tuple(map(int, set(df.fc_id)))}
        """,
        coerce_float=False,
        con=con,
    )

df = df.merge(frame_df, on=["fc_id", "frame_number"])

# %%
with bb_behavior.db.get_database_connection(application_name=APPLICATION_NAME) as con:
    detection_df = pd.read_sql(
        f"""
            SELECT x_pos, y_pos, frame_id, track_id, bee_id, detection_type, cam_id, timestamp
            FROM {bb_behavior.db.get_detections_tablename()}
            WHERE frame_id IN {tuple(map(int, set(df.frame_id)))}
        """,
        coerce_float=False,
        con=con,
    )


# %%


def line_distance(line_points, point):
    p1, p2 = line_points
    dp = p2 - p1
    reference_point = p1 + 0.3 * dp

    # return np.linalg.norm(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)

    return np.linalg.norm(point - reference_point)


# %%
detection_df.frame_id = detection_df.frame_id.astype(np.uint)
df.frame_id = df.frame_id.astype(np.uint)

# %%
distances = []
closest_detections = []
for _, datapoint in progress_bar(df.iterrows(), total=len(df)):
    line_start = np.array((datapoint.x_start, datapoint.y_start))
    line_end = np.array((datapoint.x_end, datapoint.y_end))
    line_points = line_start, line_end

    detection_subset = detection_df[
        (detection_df.frame_id == datapoint.frame_id)
        & (detection_df.detection_type == bb_tracking.types.DetectionType.TaggedBee.value)
    ]
    line_distances = detection_subset.apply(
        lambda r: line_distance(line_points, np.array((r.x_pos, r.y_pos))).astype(np.float64),
        axis=1,
    )
    closest_detection = detection_subset.iloc[line_distances.argmin()]

    distances.append(line_distances.min())
    closest_detections.append((closest_detection, datapoint))

# %%
sns.displot(distances, kde=True)
plt.xlabel("Distance (px) to GT line")
# plt.xlim([0, plt.xlim()[1]])

# %%
video_root = pathlib.Path("/home/ben/ssh/aglandgraf/beesbook/2019/hd_recording/")
video_manager = bb_behavior.io.videos.BeesbookVideoManager(
    video_root=str(video_root),
    cache_path="/home/ben/tmp/beesbook_cache/",
    videos_in_subdirectories=True,
)

# %%
"""
w, h = 128, 128
for idx in np.argsort(distances):
    detection, label = closest_detections[idx]

    frame = video_manager.get_frames([detection.frame_id])[0]

    rr, cc, val = skimage.draw.line_aa(
        int(label.y_start), int(label.x_start), int(label.y_end), int(label.x_end)
    )
    frame[rr, cc] = val

    roi = frame[
        detection.y_pos - w: detection.y_pos + w, detection.x_pos - h: detection.x_pos + h
    ]

    plt.imshow(roi, cmap=plt.cm.gray)
    plt.title(distances[idx])
    plt.show()
"""

# %%
np.sum(np.array(distances) < 35)

# %%
threshold = 35
ventilation_df = pd.concat(
    [pd.concat((e[0], e[1])) for d, e in zip(distances, closest_detections) if d < threshold],
    axis=1,
).T
# TODO: random duration
ventilation_df["from"] = ventilation_df["timestamp"]
ventilation_df["to"] = ventilation_df["timestamp"] + datetime.timedelta(seconds=2)
ventilation_df.head()

# %%
frequent_ventilation_intervals = unsupervised_behaviors.data.get_frequent_intervals(ventilation_df)

num_frames_around = 16
min_proportion_detected = 0.75

trajectory_generator = unsupervised_behaviors.data.get_event_detection_trajectory_generator(
    ventilation_df, frequent_ventilation_intervals.index, num_frames_around, min_proportion_detected
)

# %%
num_videos_total = 200
num_videos_per_behavior = 200

output_path = f"/home/ben/ssh/snuffles-data/public/benwild/predictive/videos_2019_{num_videos_total}videos_{num_frames_around}frames_ventilation.h5"
unsupervised_behaviors.data.create_video_h5(output_path, num_videos_total, num_frames_around)

# %%
groups = unsupervised_behaviors.data.load_and_store_videos(
    output_path,
    trajectory_generator,
    Behaviors.VENTILATING.value,
    0,
    num_videos_per_behavior,
    video_manager,
    use_clahe=True,
    egocentric=True,
    n_jobs=8,
)

# %%
with h5py.File(output_path, "r") as f:
    for video_idx in range(len(f["images"])):
        frames = f["images"][video_idx]
        plt.imshow(frames[0], cmap=plt.cm.gray)
        plt.show()

# %%
unsupervised_behaviors.data.extract_video(output_path, -30, "/home/ben/test.mp4", with_mask=True)
