# %%
import os
import sys

import madgrad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from unsupervised_behaviors.cpc.model import ConvCPC

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"

# %%
devices = list(range(torch.cuda.device_count()))
device = f"cuda:{devices[0]}"

# %%
latents = torch.load(
    "/srv/data/benwild/data/unsupervised_behaviors/latents--videos_2019_25000videos_32frames_random--random_image_cpc_20210921.pt"
)
model_path = "/srv/data/benwild/data/unsupervised_behaviors/cpc_on_cpc_reps_20210923.pt"

# %%
data = torch.utils.data.TensorDataset(torch.from_numpy(latents.astype(np.float32)).transpose(1, 2))
batch_size = 512

# %%
train_data, val_data = torch.utils.data.random_split(
    data, [int(0.9 * len(data)), int(0.1 * len(data))]
)

# %%
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)
val_data_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)

# %%
num_samples, num_timesteps, num_features = latents.shape

num_embeddings = 256
num_context = 256
num_ahead = 48
num_ahead_subsampling = 2

learning_rate = 0.001
weight_decay = 1e-5

num_batches = 10000

# %%
model = ConvCPC(
    num_features,
    num_embeddings,
    num_context,
    num_ahead,
    num_ahead_subsampling,
    subsample_length=num_timesteps,
    embedder_params={"num_residual_blocks_pre": 6, "num_residual_blocks": 0},
    contexter_params={"num_residual_blocks": 4, "kernel_size": 3},
).to(device)
model_parallel = torch.nn.DataParallel(model, device_ids=devices)
optimizer = madgrad.MADGRAD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%
train_losses = []
val_losses = []

# %%
def load_batch(data_loader):
    dataloader_iterator = iter(data_loader)
    while True:
        try:
            X = next(dataloader_iterator)[0]
        except StopIteration:
            dataloader_iterator = iter(data_loader)
            X = next(dataloader_iterator)[0]
        yield X


def run_batch(data_generator):
    X = next(data_generator)

    X_emb, X_ctx = model_parallel(X)
    batch_loss = model.cpc_loss(X_emb, X_ctx)

    return batch_loss


# %%
train_generator = load_batch(train_data_loader)
val_generator = load_batch(val_data_loader)

print()

for _ in range(num_batches - len(train_losses)):
    optimizer.zero_grad()

    with torch.no_grad():
        val_loss = run_batch(val_generator)

    train_loss = run_batch(train_generator)
    # scaler.scale(train_loss).backward()
    train_loss.backward()
    # scaler.unscale_(optimizer)
    optimizer.step()

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    sys.stdout.write(
        f"\r{len(train_losses) / num_batches * 100:.1f}%"
        + f" => CPC (train): {np.mean(train_losses[-100:]):.3f}"
        + f" => CPC (val): {np.mean(val_losses[-100:]):.3f}"
    )

# %%
plt.plot(pd.Series(train_losses).rolling(128).mean())
plt.plot(pd.Series(val_losses).rolling(128).mean())

# %%
torch.save((model, optimizer, train_losses, val_losses), model_path)
model_path
