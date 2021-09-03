# %%
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.dummy
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import torch
from openTSNE import TSNE

from unsupervised_behaviors.constants import DanceLabels

# %%
batch_size = 128
device = "cuda:2"

latents_path = "/srv/data/benwild/data/unsupervised_behaviors/latents.pt"
videos_path = "/storage/mi/jennyonline/data/videos_2019_10000.h5"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/cpc_20210903.pt"

# %%
latents = torch.load(latents_path)

with h5py.File(videos_path, "r") as f:
    labels = f["labels"][:]

model, _, losses = torch.load(model_path)
model = model.to(device)
plt.plot(pd.Series(losses).rolling(128).mean())

# %%
reps = model.get_representations(
    latents, batch_size, device  # , aggfunc=lambda ctx: ctx[:, :, ctx.shape[2] // 2]
)
reps.shape

# %%
embedding = TSNE(n_jobs=-1).fit(reps)

# %%
plt.scatter(embedding[:, 0], embedding[:, 1], s=1)

# %%
plt.figure(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(DanceLabels))

for label in DanceLabels:
    elems = embedding[labels == label.value]
    scatter = plt.scatter(elems[:, 0], elems[:, 1], s=3, c=[colors[label.value]], label=label.name)

plt.title("HVAE -> CPC -> TSNE")

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
    reps,
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
