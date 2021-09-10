# %%
import sys

import madgrad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import torch
import torchvision
import umap
from shared.plotting import setup_matplotlib

from unsupervised_behaviors.constants import DanceLabels
from unsupervised_behaviors.cpc.model import ImageConvCPC
from unsupervised_behaviors.data import MaskedFrameDataset

setup_matplotlib()


# %%
device = "cuda:2"

# %%
videos_path = "/storage/mi/jennyonline/data/videos_2019_10000.h5"
frame_path = "/srv/data/benwild/predictive/data/data_2020_100000_unbiased.h5"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/image_cpc_20210907.pt"

# %%
data = MaskedFrameDataset(videos_path, masked=True, horizontal_crop=24)

# %%
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data.file["mean"][()], data.file["std"][()]),
    ]
)

data = MaskedFrameDataset(
    videos_path, masked=True, transform=transforms, target_transform=lambda l: l, horizontal_crop=24
)

# %%
num_samples = len(data)

num_embeddings = 128
num_context = 128
num_ahead = 16
num_ahead_subsampling = 1
num_image_residual_blocks = 3
num_image_channels = 1

learning_rate = 0.001
weight_decay = 1e-5

tile_size = 16

num_batches = 100000
batch_size = 32

# %%
with torch.no_grad():
    sample_batch = data[0][0][None, :, :, :]

    sample_batch_tiles = torch.nn.functional.unfold(
        sample_batch, (tile_size, tile_size), stride=tile_size // 2
    )

    num_timesteps = sample_batch_tiles.shape[-1]
    num_features = num_embeddings  # sample_batch_tiles.shape[1]
    # assert num_features == tile_size ** 2

# %%
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)

# %%
model = ImageConvCPC(
    num_image_channels,
    num_features,
    num_image_residual_blocks,
    tile_size=tile_size,
    num_embeddings=num_embeddings,
    num_context=num_context,
    num_ahead=num_ahead,
    num_ahead_subsampling=num_ahead_subsampling,
    subsample_length=num_timesteps,
    embedder_params={"num_residual_blocks_pre": 6, "num_residual_blocks": 0},
    contexter_params={"num_residual_blocks": 4, "kernel_size": 3},
).to(device)

optimizer = madgrad.MADGRAD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%
losses = []

# %%
print()
dataloader_iterator = iter(data_loader)
for _ in range(num_batches - len(losses)):
    optimizer.zero_grad()

    try:
        X, target = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(data_loader)
        X, target = next(dataloader_iterator)

    X = X.to(device, non_blocking=True)

    X_emb, X_ctx = model(X)

    batch_loss = model.cpc_loss(X_emb, X_ctx)

    batch_loss.backward()
    optimizer.step()

    losses.append(batch_loss.item())
    sys.stdout.write(f"\r{len(losses) / num_batches} - {np.mean(losses[-100:]):.3f}")

# %%
plt.plot(pd.Series(losses).rolling(1024).mean())

# %%
torch.multiprocessing.set_sharing_strategy("file_system")

# %%
cpc_reps = []
labels = []

data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)

with torch.no_grad():
    for x, y in data_loader:
        X = x.to(device, non_blocking=True)
        X_emb, X_ctx = model(X)

        cpc_reps.append(X_ctx.mean(dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())

cpc_reps = np.concatenate(cpc_reps)
labels = np.concatenate(labels)

# %%
cpc_reps.shape

# %%
embedding = umap.UMAP(
    n_components=2,
    n_jobs=-1,
).fit_transform(cpc_reps)

# %%
plt.figure(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(DanceLabels))

for label in DanceLabels:
    elems = embedding[labels == label.value]
    scatter = plt.scatter(elems[:, 0], elems[:, 1], s=3, c=[colors[label.value]], label=label.name)

plt.title("CPC -> UMAP")
plt.legend()

# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000, n_jobs=4)

sklearn.model_selection.cross_val_score(
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
torch.save((model, optimizer, losses), model_path)
