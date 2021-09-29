# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from fastprogress.fastprogress import force_console_behavior

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
config.videos_path = "/storage/mi/jennyonline/data/videos_2019_10000.h5"
config.model_path = "/srv/data/benwild/data/unsupervised_behaviors/image_cpc_20210914.pt"

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

devices = (0, 1, 2, 3)
device = f"cuda:{devices[0]}"
config.batch_size = 48 * len(devices)

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
    data, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

with torch.no_grad():
    for X, y in progress_bar(data_loader, total=len(data_loader)):
        X_emb, X_ctx = model_parallel(X)
        clf_input = X_emb.reshape(X.shape[0], num_timesteps, config.num_embeddings, num_timesteps)
        clf_input = clf_input.transpose(2, 1)
        clf_input = torch.nn.functional.normalize(clf_input, dim=1, p=2)
        clf_input = clf_input.mean(dim=(2, 3))

        # TODO: save as hdf5
        cpc_reps.append(clf_input.half().cpu().numpy())
        labels.append(y.cpu().numpy())

cpc_reps = np.concatenate(cpc_reps)
labels = np.concatenate(labels)


# %%
cpc_reps.shape

# %%
cpc_reps = cpc_reps.reshape(-1, 33, config.num_embeddings)
cpc_reps.shape

# %%
videos_fname = config.videos_path.split("/")[-1].split(".")[0]
model_fname = config.model_path.split("/")[-1].split(".")[0]

config.latents_path = (
    f"/srv/data/benwild/data/unsupervised_behaviors/latents--{videos_fname}--{model_fname}.pt"
)
config.latents_path

# %%
torch.save(cpc_reps, config.latents_path)
