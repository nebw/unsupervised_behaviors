# %%
import os
import sys

import cloudpickle
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
import wandb

from unsupervised_behaviors.constants import Behaviors
from unsupervised_behaviors.cpc.model import ColumnImageConvCPC
from unsupervised_behaviors.data import MaskedFrameDataset

from shared.plotting import setup_matplotlib

setup_matplotlib()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# %%
wandb.init(project="unsupervised_behaviors", entity="nebw")
config = wandb.config

# %%
config.videos_path = "/storage/mi/jennyonline/data/videos_2019_10000.h5"
# config.frame_path = "/srv/data/benwild/predictive/data/data_2020_100000_unbiased.h5"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/image_cpc_20210916.pt"

# %%
data = MaskedFrameDataset(config.videos_path)

# %%
idxs = sorted(np.random.choice(len(data), size=1000))
samples = np.stack([data[i][0] for i in idxs])

# %%
assert len(np.array(idxs)[np.argwhere(samples.sum(axis=(1, 2, 3)) == 0.0)[:, 0]]) == 0

# %%
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
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

devices = 0
device = f"cuda:{devices[0]}"
config.batch_size = 18 * len(devices)

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
train_data, val_data = torch.utils.data.random_split(
    data, [int(0.85 * len(data)), int(0.15 * len(data))]
)

train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
)
val_data_loader = torch.utils.data.DataLoader(
    val_data, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True
)


# %%
model = ColumnImageConvCPC(
    num_image_channels=config.num_image_channels,
    num_features=num_features,
    num_image_residual_blocks=config.num_image_residual_blocks,
    tile_size=config.tile_size,
    num_embeddings=config.num_embeddings,
    num_context=config.num_context,
    num_ahead=config.num_ahead,
    num_ahead_subsampling=config.num_ahead_subsampling,
    subsample_length=num_timesteps,
    embedder_params={"num_residual_blocks_pre": 6, "num_residual_blocks": 0},
    contexter_params={"num_residual_blocks": 4, "kernel_size": 3},
).to(device)

# model_parallel = torch.nn.DataParallel(model, device_ids=devices)
model_parallel = model
wandb.watch(model_parallel)

optimizer = madgrad.MADGRAD(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scaler = torch.cuda.amp.GradScaler()

# %%
train_losses = []
val_losses = []

# %%
def load_batch(data_loader):
    dataloader_iterator = iter(data_loader)
    while True:
        try:
            X, _ = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(data_loader)
            X, _ = next(dataloader_iterator)
        X = X.to(device, non_blocking=True)
        yield X


def run_batch(data_generator):
    X = next(data_generator)

    with torch.cuda.amp.autocast():
        X_emb, X_ctx = model_parallel(X)
        batch_loss = model.cpc_loss(X_emb, X_ctx)

    return batch_loss


# %%
train_generator = load_batch(train_data_loader)
val_generator = load_batch(val_data_loader)
print()
for _ in range(config.num_batches - len(train_losses)):
    optimizer.zero_grad()

    with torch.no_grad():
        val_loss = run_batch(val_generator)

    train_loss = run_batch(train_generator)
    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    wandb.log({"loss": train_loss, "val_loss": val_loss})

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    sys.stdout.write(
        f"\r{len(train_losses) / config.num_batches * 100:.1f}%"
        + f" => CPC (train): {np.mean(train_losses[-100:]):.3f}"
        + f" => CPC (val): {np.mean(val_losses[-100:]):.3f}"
    )

# %%
plt.plot(pd.Series(train_losses).rolling(128).mean())
plt.plot(pd.Series(val_losses).rolling(128).mean())

# %%
torch.multiprocessing.set_sharing_strategy("file_system")

# %%
cpc_reps = []
labels = []

data_loader = torch.utils.data.DataLoader(
    data, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

with torch.no_grad():
    for x, y in data_loader:
        X = x.to(device, non_blocking=True)
        X_emb, X_ctx = model_parallel(X)

        clf_input = X_emb.reshape(X.shape[0], num_timesteps, config.num_embeddings, num_timesteps)
        clf_input = clf_input.transpose(2, 1)
        clf_input = clf_input.mean(dim=(2, 3))

        cpc_reps.append(clf_input.cpu().numpy())
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
fig, ax = plt.subplots(figsize=(12, 6))

colors = sns.color_palette(n_colors=len(Behaviors))

for label in Behaviors:
    elems = embedding[labels == label.value]
    if len(elems) > 0:
        scatter = plt.scatter(
            elems[:, 0], elems[:, 1], s=10, c=[colors[label.value]], label=label.name, alpha=0.25
        )

plt.title("CPC -> UMAP")
plt.legend()

wandb.log({"Embedding (UMAP)": fig})

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

wandb.log({"ROC AUC Score (Linear Classifier)": score})

# %%
torch.save((model, optimizer, train_losses, val_losses), model_path, pickle_module=cloudpickle)
