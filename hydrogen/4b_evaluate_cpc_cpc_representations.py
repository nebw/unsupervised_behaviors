# %%
import itertools
import os
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.decomposition
import sklearn.dummy
import sklearn.linear_model
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection
import torch
import umap
from fastprogress.fastprogress import force_console_behavior
from openTSNE import TSNE

import bb_behavior
import bb_behavior.db
from bb_behavior.trajectory.features import FeatureTransform

import unsupervised_behaviors.data
from unsupervised_behaviors.baselines import FrameCNNBaseline
from unsupervised_behaviors.constants import Behaviors

from shared.plotting import setup_matplotlib

master_bar, progress_bar = force_console_behavior()
setup_matplotlib()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

APPLICATION_NAME = "unsupervised_behaviors"

bb_behavior.db.set_season_berlin_2019()

# %%
batch_size = 128
device = "cuda:0"

latents_path = "/srv/data/benwild/data/unsupervised_behaviors/latents--videos_2019_20000videos_32frames_allbehaviors--image_cpc_20210917.pt"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/cpc_on_cpc_reps_20210921.pt"
videos_path = "/srv/public/benwild/predictive/videos_2019_20000videos_32frames_allbehaviors.h5"

# %%
with h5py.File(videos_path, "r") as f:
    labels = f["labels"][:]

# %%
with h5py.File(videos_path, "r") as f:
    frame_ids = f["frame_ids"][:]
    center_idx = frame_ids.shape[1] // 2
    frame_ids = frame_ids[:, center_idx]
    x_pos = f["x_pos"][:, center_idx].astype(np.uint64)
    y_pos = f["y_pos"][:, center_idx].astype(np.uint64)

truth_df = pd.DataFrame(
    np.stack((frame_ids, x_pos, y_pos)).T, columns=["frame_id", "x_pos", "y_pos"]
)

# %%
detections = bb_behavior.db.sampling.get_detections_dataframe_for_frames(
    truth_df.frame_id, use_hive_coordinates=False
)
detections.frame_id = detections.frame_id.astype(np.uint64)
dets_by_frame_id = dict(list(detections.groupby("frame_id")))

# %%


def get_matching_bee_id(row):
    df = dets_by_frame_id[row.frame_id]
    p0 = np.array((row.x_pos, row.y_pos))[None, :]
    p1 = np.stack((df.x_pos, df.y_pos)).T
    distances = np.linalg.norm(p0 - p1, ord=2, axis=1)
    return df.iloc[np.argmin(distances)].bee_id


bee_ids = truth_df.apply(get_matching_bee_id, axis=1)
truth_df["bee_id"] = bee_ids
truth_df["label"] = labels

# %%
features = [FeatureTransform.Angle2Geometric(), FeatureTransform.Egomotion()]

data_reader = bb_behavior.trajectory.features.DataReader(
    dataframe=truth_df,
    use_hive_coords=True,
    frame_margin=8,
    target_column="label",
    feature_procs=features,
    sample_count=None,
    chunk_frame_id_queries=True,
    n_threads=4,
)

data_reader.create_features()

traj_X = sklearn.decomposition.PCA(n_components=0.99).fit_transform(
    data_reader.X.reshape(len(data_reader.X), -1)
)
traj_Y = data_reader.Y.astype(int)[:, 0]

# %%
frame_idx = 32
with h5py.File(videos_path, "r") as f:
    video_idx = sorted(np.random.choice(np.arange(len(f["images"])), size=1000, replace=False))
    video = f["images"][video_idx, frame_idx]
    mask = f["tag_masks"][video_idx, frame_idx] * f["loss_masks"][video_idx, frame_idx]
    frames = video * mask


# %%
frames_pca = sklearn.decomposition.PCA(n_components=0.99).fit(frames.reshape(len(frames), -1))
frames_pca.transform(frames.reshape(len(frames), -1)).shape

# %%
frame_reps = []
with h5py.File(videos_path, "r") as f:
    i = 0
    while i < len(f["images"]):
        video_idx = np.arange(i, min(len(f["images"]), i + batch_size))

        video = f["images"][video_idx, frame_idx]
        mask = f["tag_masks"][video_idx, frame_idx] * f["loss_masks"][video_idx, frame_idx]
        frames = video * mask

        frame_reps.append(frames_pca.transform(frames.reshape(len(frames), -1)))

        i += batch_size

frame_reps = np.concatenate(frame_reps)
frame_reps.shape

# %%
with h5py.File(videos_path, "r") as f:
    frames = f["images"][:, frame_idx]
    mask = f["tag_masks"][:, frame_idx] * f["loss_masks"][:, frame_idx]
    frames = frames * mask
    del mask

frames = frames.astype(np.float32) / 255
frames = frames[:, None, :, :]

# %%
latents = torch.load(latents_path)

model, _, train_losses, val_losses = torch.load(model_path)
model = model.to(device)
plt.plot(pd.Series(train_losses).rolling(128).mean())
plt.plot(pd.Series(val_losses).rolling(128).mean())

# %%
reps = model.get_representations(
    latents,
    batch_size,
    device,
    # aggfunc=lambda ctx: torch.nn.functional.normalize(ctx[:, :, ctx.shape[2] // 2], dim=1, p=2),
    # aggfunc=lambda ctx: torch.nn.functional.normalize(ctx, dim=1, p=2).mean(dim=-1),
    # aggfunc=lambda ctx: ctx[:, :, ctx.shape[2] // 2 - 8 : ctx.shape[2] // 2 + 8].mean(dim=-1),
    aggfunc=lambda ctx: ctx.mean(dim=-1),
)
reps.shape

# %%
reps_pca = sklearn.decomposition.PCA(n_components=0.999).fit_transform(reps)
reps_pca.shape

# %%
embedding = TSNE(n_jobs=-1).fit(reps_pca)

# %%
plt.scatter(embedding[:, 0], embedding[:, 1], s=1)

# %%
plt.figure(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(Behaviors))

for label in Behaviors:
    elems = embedding[labels == label.value]
    scatter = plt.scatter(elems[:, 0], elems[:, 1], s=3, c=[colors[label.value]], label=label.name)

plt.title("Image-CPC -> CPC -> TSNE")

plt.legend()

# %%
linear = sklearn.linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000, n_jobs=4, class_weight="balanced"
)


def get_scores(X, y, n_splits=12, n_jobs=-1, model=linear, train_size=None):
    return sklearn.model_selection.cross_val_score(
        model,
        X,
        y,
        cv=sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size),
        scoring=sklearn.metrics.make_scorer(
            sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
        ),
        n_jobs=n_jobs,
    ).mean()


# %%
get_scores(latents.mean(axis=1), labels).mean()

# %%
get_scores(reps, labels).mean()

# %%
get_scores(frame_reps, labels).mean()

# %%
get_scores(frames, labels, n_jobs=4, model=FrameCNNBaseline(labels, device)).mean()

# %%
get_scores(traj_X, traj_Y).mean()

# %%
get_scores(reps, labels, model=sklearn.dummy.DummyClassifier()).mean()


# %%
num_samples = []
scores_latents = []
scores_cpc = []
scores_frames = []
scores_supervised = []
scores_trajs = []
for train_size in progress_bar(
    np.logspace(1, np.log10(len(frame_reps) - 500), num=20).astype(np.int)
):
    score = get_scores(frame_reps, labels, train_size=train_size).mean()
    scores_frames.append(score)

    score = get_scores(
        frames, labels, n_jobs=4, model=FrameCNNBaseline(labels, device), train_size=train_size
    ).mean()
    scores_supervised.append(score)

    score = get_scores(reps, labels, train_size=train_size).mean()
    scores_cpc.append(score)

    score = get_scores(latents.mean(axis=1), labels, train_size=train_size).mean()
    scores_latents.append(score)

    score = get_scores(traj_X, traj_Y, train_size=train_size).mean()
    scores_trajs.append(score)

    num_samples.append(train_size)

# %%
plt.figure(figsize=(12, 6))
plt.plot(num_samples, scores_cpc, label="CPC Representations")
plt.plot(num_samples, scores_latents, label="Center Frame (Image CPC)")
plt.plot(num_samples, scores_supervised, label="Center Frame (Supervised CNN)")
plt.plot(num_samples, scores_trajs, label="Trajectory features PCA")
plt.plot(num_samples, scores_frames, label="Center Frame (Pixel PCA)")
plt.xlabel("Number of training samples")
plt.ylabel("Mean ROC AUC (CV)")
plt.title("Multinomial regression [Dance, Following, Ventilating, Unknown]")
plt.grid()
plt.xlim([min(num_samples), max(num_samples)])
plt.ylim([0.5, 1])
plt.legend()

# %%
unknown_idxs = labels == 0
unknown_reps = reps[unknown_idxs]

# %%
mixture = sklearn.mixture.BayesianGaussianMixture(n_components=6)
mixture.fit(unknown_reps)


# %%
cluster_probs = mixture.predict_proba(unknown_reps)
clusters = mixture.predict(unknown_reps)
cluster_probs = np.stack([cluster_probs[idx, i] for idx, i in enumerate(clusters)])

np.unique(clusters, return_counts=True)

# %%
reps_2d = umap.UMAP(n_jobs=-1).fit_transform(unknown_reps)

# %%
plt.figure(figsize=(12, 8))
plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=clusters, cmap="Spectral", s=cluster_probs * 5)

# %%
original_idxs = np.argwhere(unknown_idxs)[:, 0]

for cluster in np.arange(0, clusters.max() + 1):
    print(cluster)

    mean_rep = unknown_reps[clusters == cluster].mean(axis=0)
    most_similar_idxs = np.linalg.norm(unknown_reps - mean_rep[None, :], axis=1).argsort()
    random_similar_idxs = np.random.choice(most_similar_idxs[:100], size=6)

    paths = []
    for i in range(6):
        idx = original_idxs[most_similar_idxs[i]]
        path = f"/srv/public/benwild/cluster_{cluster}_{i}.mp4"
        unsupervised_behaviors.data.extract_video(videos_path, idx, path)
        paths.append(path)

    filter_str = "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[v]"
    subprocess.run(
        ["ffmpeg", "-y"]
        + list(itertools.chain.from_iterable([["-i", path] for path in paths]))
        + [
            "-filter_complex",
            filter_str,
            "-map",
            "[v]",
            f"/srv/public/benwild/cluster_{cluster}_grid.mp4",
        ]
    )
    subprocess.run(
        ["ffmpeg", "-y"]
        + [
            "-i",
            f"/srv/public/benwild/cluster_{cluster}_grid.mp4",
            "-vf",
            "fps=6,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            f"/srv/public/benwild/cluster_{cluster}_grid.gif",
        ]
    )
