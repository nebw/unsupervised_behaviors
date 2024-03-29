# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import torch
import torchvision
import umap
from fastprogress.fastprogress import force_console_behavior

from unsupervised_behaviors.constants import Behaviors
from unsupervised_behaviors.data import MaskedFrameDataset

from shared.plotting import setup_matplotlib

master_bar, progress_bar = force_console_behavior()
setup_matplotlib()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# %%


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = DotDict()


# %%
config.videos_path = (
    "/srv/public/benwild/predictive/videos_2019_5000videos_32frames_allbehaviors_fixed.h5"
)
config.model_path = "/srv/data/benwild/data/unsupervised_behaviors/random_image_cpc_20210921.pt"

# %%
data = MaskedFrameDataset(config.videos_path)

# %%
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data.file["mean"][()], data.file["std"][()]),
    ]
)

data = MaskedFrameDataset(
    config.videos_path,
    ellipse_masked=False,
    tag_masked=True,
    transform=transforms,
    target_transform=lambda l: l,
    horizontal_crop=None,
)
plt.imshow((data[0][0].permute(1, 2, 0)), cmap=plt.cm.gray)

# %%
config.num_samples = len(data)

config.num_embeddings = 128
config.num_context = 128
config.num_ahead = 5
config.num_ahead_subsampling = 1
config.num_image_residual_blocks = 7
config.num_image_channels = 1

config.learning_rate = 0.001
config.weight_decay = 1e-5

config.tile_size = 32

devices = list(range(torch.cuda.device_count()))
device = f"cuda:{devices[0]}"
config.batch_size = 16 * len(devices)

config.num_batches = 20000

# %%
with torch.no_grad():
    sample_batch = data[0][0][None, :, :, :]

    sample_batch_tiles = torch.nn.functional.unfold(
        sample_batch, (config.tile_size, config.tile_size), stride=config.tile_size // 2
    )

    num_timesteps = int(np.sqrt(sample_batch_tiles.shape[-1]))
    num_features = config.num_embeddings

    print(num_timesteps)

# %%
model, optimizer, train_losses, val_losses = torch.load(config.model_path)
model = model.to(device)
model_parallel = torch.nn.DataParallel(model, device_ids=devices)

# %%
plt.plot(pd.Series(train_losses).rolling(128).mean())
plt.plot(pd.Series(val_losses).rolling(128).mean())

# %%
torch.multiprocessing.set_sharing_strategy("file_system")

# %%
cpc_reps = []
labels = []

data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=MaskedFrameDataset.CenterFrameSampler(data, 65),
)

with torch.no_grad():
    for X, y in progress_bar(data_loader, total=len(data_loader)):
        X_emb, X_ctx = model_parallel(X)

        clf_input = X_emb.reshape(X.shape[0], num_timesteps, config.num_embeddings, num_timesteps)
        clf_input = clf_input.transpose(2, 1)
        clf_input = torch.nn.functional.normalize(clf_input, dim=1, p=2)
        clf_input = clf_input.mean(dim=(2, 3))

        cpc_reps.append(clf_input.float().cpu().numpy())
        labels.append(y.cpu().numpy())

cpc_reps = np.concatenate(cpc_reps)
labels = np.concatenate(labels)

# %%
cpc_reps.shape

# %%
embedding = umap.UMAP(n_components=2, n_jobs=-1, output_metric="haversine").fit_transform(cpc_reps)

# %%
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})

colors = sns.color_palette(n_colors=len(Behaviors))

for label in Behaviors:
    elems = embedding[labels == label.value]

    x = np.sin(elems[:, 0]) * np.cos(elems[:, 1])
    y = np.sin(elems[:, 0]) * np.sin(elems[:, 1])
    z = np.cos(elems[:, 0])

    if len(elems) > 0:
        scatter = ax.scatter(x, y, z, s=10, c=[colors[label.value]], label=label.name, alpha=0.5)

plt.title("CPC -> UMAP")
plt.legend()


# %%
fig, ax = plt.subplots(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(Behaviors))

for label in Behaviors:
    elems = embedding[labels == label.value]

    x = np.sin(elems[:, 0]) * np.cos(elems[:, 1])
    y = np.sin(elems[:, 0]) * np.sin(elems[:, 1])
    z = np.cos(elems[:, 0])

    x = np.arctan2(x, y)
    y = -np.arccos(z)

    if len(elems) > 0:
        scatter = ax.scatter(x, y, s=10, c=[colors[label.value]], label=label.name, alpha=0.5)

plt.title("CPC -> UMAP")
plt.legend()


# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000, n_jobs=4)

score = sklearn.model_selection.cross_val_score(
    linear,
    cpc_reps,
    labels,
    cv=sklearn.model_selection.StratifiedShuffleSplit(),
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.roc_auc_score, multi_class="ovo", needs_proba=True
    ),
    n_jobs=-1,
).mean()

# %%
score
