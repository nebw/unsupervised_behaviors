# %%
import itertools
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
from openTSNE import TSNE

import unsupervised_behaviors.data
from unsupervised_behaviors.constants import Behaviors

from shared.plotting import setup_matplotlib

setup_matplotlib()

# %%
batch_size = 128
device = "cuda:2"

latents_path = "/srv/data/benwild/data/unsupervised_behaviors/latents--videos_2019_10000--image_cpc_20210914.pt"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/cpc_on_cpc_reps_20210915.pt"
videos_path = "/storage/mi/jennyonline/data/videos_2019_10000.h5"

# %%
frame_idx = 16
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
latents = torch.load(latents_path)

with h5py.File(videos_path, "r") as f:
    labels = f["labels"][:]

model, _, losses = torch.load(model_path)
model = model.to(device)
plt.plot(pd.Series(losses).rolling(1024).mean())

# %%
reps = model.get_representations(
    latents, batch_size, device  # , aggfunc=lambda ctx: ctx[:, :, ctx.shape[2] // 2]
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
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000, n_jobs=4)
sklearn.model_selection.cross_val_score(
    linear,
    latents.mean(axis=1),
    labels,
    cv=sklearn.model_selection.StratifiedShuffleSplit(),
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
    ),
    n_jobs=-1,
).mean()

# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000, n_jobs=4)
sklearn.model_selection.cross_val_score(
    linear,
    reps_pca,
    labels,
    cv=sklearn.model_selection.StratifiedShuffleSplit(),
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
    ),
    n_jobs=-1,
).mean()

# %%
sklearn.model_selection.cross_val_score(
    sklearn.dummy.DummyClassifier(),
    reps,
    labels,
    cv=sklearn.model_selection.StratifiedShuffleSplit(),
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
    ),
).mean()
# %%

# %%
num_samples = []
scores_latents = []
scores_cpc = []
scores_frames = []
for train_size in np.logspace(1, np.log10(9500), num=15).astype(np.int):
    score = sklearn.model_selection.cross_val_score(
        linear,
        frame_reps,
        labels,
        cv=sklearn.model_selection.StratifiedShuffleSplit(train_size=train_size, n_splits=25),
        scoring=sklearn.metrics.make_scorer(
            sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
        ),
        n_jobs=-1,
    ).mean()
    scores_frames.append(score)

    score = sklearn.model_selection.cross_val_score(
        linear,
        reps_pca,
        labels,
        cv=sklearn.model_selection.StratifiedShuffleSplit(train_size=train_size, n_splits=25),
        scoring=sklearn.metrics.make_scorer(
            sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
        ),
        n_jobs=-1,
    ).mean()
    scores_cpc.append(score)

    score = sklearn.model_selection.cross_val_score(
        linear,
        latents.mean(axis=1),
        labels,
        cv=sklearn.model_selection.StratifiedShuffleSplit(train_size=train_size, n_splits=25),
        scoring=sklearn.metrics.make_scorer(
            sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
        ),
        n_jobs=-1,
    ).mean()
    scores_latents.append(score)

    num_samples.append(train_size)

# %%
plt.figure(figsize=(12, 6))
plt.plot(num_samples, scores_frames, label="Center Frame (Pixel PCA)")
plt.plot(num_samples, scores_latents, label="HVAE Latents")
plt.plot(num_samples, scores_cpc, label="CPC Representations")
plt.xlabel("Number of training samples")
plt.ylabel("Mean ROC AUC (CV)")
plt.title("Multinomial regression [Dance, Following, Unknown]")
plt.grid()
plt.xlim([min(num_samples), max(num_samples)])
plt.legend()

import hdbscan
import sklearn.cluster as cluster

# %%
import umap
from sklearn.metrics import adjusted_mutual_info_score

# %%
clusterable_embedding = umap.UMAP(
    n_neighbors=30, min_dist=0.0, n_components=2, n_jobs=-1
).fit_transform(reps_pca)

# %%
plt.figure(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(Behaviors))

for label in Behaviors:
    elems = clusterable_embedding[labels == label.value]
    scatter = plt.scatter(elems[:, 0], elems[:, 1], s=3, c=[colors[label.value]], label=label.name)

plt.title("Image-CPC -> CPC -> UMAP")

plt.legend()

# %%
clusters = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=50,
).fit_predict(clusterable_embedding)

# %%
clustered = clusters >= 0
plt.scatter(
    clusterable_embedding[~clustered, 0],
    clusterable_embedding[~clustered, 1],
    color=(0.5, 0.5, 0.5),
    s=0.1,
    alpha=0.5,
)

plt.scatter(
    clusterable_embedding[clustered, 0],
    clusterable_embedding[clustered, 1],
    c=clusters[clustered],
    s=0.1,
    cmap="Spectral",
)

# %%
adjusted_mutual_info_score(labels[clustered] == Behaviors.DANCE.value, clusters[clustered])

# %%
adjusted_mutual_info_score(labels[clustered] == Behaviors.FOLLOWING.value, clusters[clustered])

# %%

# %%

# %%
clusterer = sklearn.mixture.BayesianGaussianMixture(
    n_components=100, max_iter=1000, n_init=10, weight_concentration_prior=1 / 1000
)
clusters = clusterer.fit_predict(reps_pca)

np.unique(clusters, return_counts=True)

# %%
for cluster in np.arange(0, clusters.max() + 1):
    print(cluster)
    mean_rep = reps[clusters == cluster].mean(axis=0)
    most_similar_idxs = np.linalg.norm(reps - mean_rep[None, :], axis=1).argsort()

    paths = []
    for i in range(6):
        path = f"/srv/public/benwild/cluster_{cluster}_{i}.mp4"
        unsupervised_behaviors.data.extract_video(videos_path, most_similar_idxs[i], path)
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

# %%
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=clusters[:],
    s=0.1,
    cmap="Spectral",
)
