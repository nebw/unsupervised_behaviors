# %%
import sys

import madgrad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from unsupervised_behaviors.cpc.model import ConvCPC

# %%
device = "cuda:2"

# %%
latents = torch.load("/srv/data/benwild/data/unsupervised_behaviors/latents.pt")
model_path = "/srv/data/benwild/data/unsupervised_behaviors/cpc_20210903.pt"

num_samples, num_timesteps, num_features = latents.shape

num_embeddings = 256
num_context = 128
num_ahead = 16
num_ahead_subsampling = 1

learning_rate = 0.001
weight_decay = 1e-5

num_batches = 10000
batch_size = 128

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
optimizer = madgrad.MADGRAD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%
losses = []

# %%
for _ in range(num_batches - len(losses)):
    optimizer.zero_grad()

    batch_idxs = np.random.randint(0, num_samples, size=batch_size)
    X = torch.from_numpy(latents[batch_idxs]).to(device, non_blocking=True)
    X = X.transpose(1, 2)

    X_emb, X_ctx = model(X)

    batch_loss = model.cpc_loss(X_emb, X_ctx)

    batch_loss.backward()
    optimizer.step()

    losses.append(batch_loss.item())
    sys.stdout.write(f"\r{len(losses) / num_batches} - {np.mean(losses[-100:]):.3f}")

# %%
plt.plot(pd.Series(losses).rolling(128).mean())

# %%
torch.save((model, optimizer, losses), model_path)
