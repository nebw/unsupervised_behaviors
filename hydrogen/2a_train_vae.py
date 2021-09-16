# %% codecell
import io
import os

# %% codecell
import time

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm

import hierarchical_vae.hps as hps
from hierarchical_vae.train_helpers import load_opt, set_up_hyperparams
from hierarchical_vae.vae import VAE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# %% codecell
device = "cuda:0"
np.set_printoptions(threshold=5)

# %% codecell
# masks are already stored as boolean values
f = h5py.File("/srv/data/benwild/predictive/data/data_2020_100000_unbiased.h5", "r")

images = f["images"]
tag_masks = f["tag_masks"]
loss_masks = f["loss_masks"]
labels = f["labels"]

# %% codecell
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(images[0], cmap=plt.cm.gray)
axes[1].imshow(images[0] * loss_masks[0], cmap=plt.cm.gray)
axes[2].imshow(images[0] * loss_masks[0] * tag_masks[0], cmap=plt.cm.gray)

# %% codecell
H = set_up_hyperparams(s=["--dataset=i64"])
H_ = hps.ffhq_256
H_["image_channels"] = 1
H_["image_size"] = 128
H_["width"] = 128
H_["n_batch"] = 8
H_.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x20,64m32,64x12,128m64"
H_.enc_blocks = "128x4,128d2,64x7,64d2,32x7,32d2,16x7,16d2,8x7,8d2,4x7,4d4,1x8"
H_["adam_warmup_iters"] = 100
H.update(H_)
H["skip_threshold"] = -1

H.lr = 0.0001
H.num_epochs = 50

# %% codecell
vae = VAE(H).to(device)
clf = torch.nn.Linear(16, 1).to(device)

optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(
    H, vae, extra_params=list(clf.parameters())
)

clf_loss = torch.nn.BCEWithLogitsLoss().to(device)

elbos = []
bces = []
accs = []

# %% codecell
H["std"] = f["std"][()]
std = H["std"]
H["mean"] = f["mean"][()]
mean = H["mean"]

# %% codecell
# precompute per-sample probabilities s.t. 50% of randomly drawn samples have deformed wings
p_deformed = np.mean(labels)
print(p_deformed)
sample_probs = np.ones(len(labels))
sample_probs *= p_deformed
sample_probs[labels] = 1 - p_deformed

# validation set
sample_probs[-10000:] = 0.0

sample_probs /= np.sum(sample_probs)

# %%
"""
state = torch.load('/srv/data/benwild/predictive/data/vae_2020_20210702.pt')

elbos = state['elbos']
vae.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
"""

# %% codecell
start = time.time()
base = "vae_"

vae.to(device)
vae.train()

save_stats = []

for i_epoch in range(H.num_epochs):
    progress = tqdm.trange(len(images) // H_["n_batch"])

    # for every batch
    for i in progress:
        random_idxs = sorted(
            np.random.choice(np.arange(len(images)), H_["n_batch"], p=sample_probs, replace=False)
        )

        x = torch.from_numpy(images[random_idxs][:, :, :, None].astype(np.float32))
        x -= H["mean"]
        x /= H["std"]
        x = x.to(device)
        l = torch.from_numpy(labels[random_idxs].astype(np.float))
        l = l.to(device)

        target_mask = torch.from_numpy(loss_masks[random_idxs][:, :, :, None]).to(device)
        tag_mask = torch.from_numpy(tag_masks[random_idxs][:, :, :, None]).to(device)
        data_input = (x * tag_mask).float()
        target = data_input.clone().detach()

        vae.zero_grad()
        stats = vae.forward(
            data_input,
            target,
            target_mask * tag_mask,
            use_beta=True,
            beta_factor=0.1,
            get_latents=True,
        )
        z = stats["decoder_stats"][0]["z_undetached"].squeeze()
        z_loss = clf_loss(clf(z).squeeze(), l)
        acc = ((torch.sigmoid(clf(z).squeeze()) > 0.5) == l.bool()).float().mean()

        (stats["elbo"] + z_loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()

        distortion_nans = torch.isnan(stats["distortion"]).sum()
        rate_nans = torch.isnan(stats["rate"]).sum()

        stats.update(
            dict(
                rate_nans=0 if rate_nans == 0 else 1,
                distortion_nans=0 if distortion_nans == 0 else 1,
            )
        )

        elbos.append(stats["elbo"].item())
        bces.append(z_loss.item())
        accs.append(acc.item())

        # only do an update step if no rank has a NaN and if the grad norm is below a specific threshold
        if (
            stats["distortion_nans"] == 0
            and stats["rate_nans"] == 0
            and (H.skip_threshold == -1 or grad_norm < H.skip_threshold)
        ):
            optimizer.step()
            skipped_updates = 0

            progress.set_postfix(
                dict(
                    ELBO=np.nanmean(elbos[-100:]),
                    BCE=np.nanmean(bces[-100:]),
                    ACC=np.nanmean(accs[-100:]),
                    lr=scheduler.get_last_lr()[0],
                    has_nan=np.any(np.isnan(elbos[-100:])),
                )
            )

            scheduler.step()

    print("Epoch ", i_epoch, " is over")

end = time.time()
print("Runtime: ", ((end - start) / 60), " Minuten")

# %%

# %% codecell
##torch.save(vae.state_dict(),"/srv/data/benwild/predictive/data/vae2020_20210706_model_state.pt")

torch.save(
    {
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "clf_state_dict": clf.state_dict(),
        "elbos": elbos,
        "accs": accs,
        "bces": bces,
    },
    "/srv/data/benwild/predictive/data/vae_2019_20210809_semisupervised.pt",
)

# %%
# %% codecell
# compute mean via moving windows
pd.Series(elbos).rolling(1024, min_periods=200).mean()[1500:].plot()
# plt.ylim([3.0, 3.15])
# %% codecell
pd.Series(accs).rolling(1024, min_periods=200).mean()[1500:].plot()

# %%
####### get latents of images out of saved model and print images ####################
# does not work at the moment due to problems with data being on cpu/gpu

vae.cpu()
vae.eval()

# choose one image from first batch for plotting -> here first image of batch is chosen
sample_idx = 2
temperature = 0.5
min_kl = 0

x = torch.from_numpy(images[: H["n_batch"]].astype(np.float32))[:, :, :, None]
x -= mean
x /= std

tag_mask = torch.from_numpy(tag_masks[0].astype(np.float32))[None, :, :, None]
mask = (loss_masks[0] * tag_masks[0]).astype(np.float32)
data_input = (x * tag_mask).float()

fig, axes = plt.subplots(1, 7, figsize=(20, 8))

axes[0].imshow(
    ((data_input[sample_idx].data.numpy() * std) + mean)[:, :, 0],
    # grayscale image
    cmap=plt.cm.gray,
)

# get threshold for pixels by using highest and lowest value in data
minv = ((data_input[sample_idx].data.numpy() * std) + mean)[:, :, 0].min()
maxv = ((data_input[sample_idx].data.numpy() * std) + mean)[:, :, 0].max()

with io.BytesIO() as f:
    imageio.imsave(
        f, ((data_input[sample_idx] * std) + mean).data.numpy().astype(np.uint8), format="png"
    )
    f.flush()
    f.seek(0)
    bytes_png = len(f.read())

axes[0].set_title(f"$x$ - {bytes_png / 1024:.3f}KiB (PNG)")

with torch.no_grad():
    zs = [s["z"] for s in vae.forward_get_latents(data_input)]
    kls = [s["kl"] for s in vae.forward_get_latents(data_input)]

    for z, k in zip(zs, kls):
        z[k < min_kl] = 0
        k[k < min_kl] = 0

    qms = [s["qm"] for s in vae.forward_get_latents(data_input)]
    qvs = [s["qv"] for s in vae.forward_get_latents(data_input)]

    # currently: mb=4 -> number of images in a batch
    mb = data_input.shape[0]


def plot_layer(ax, layer_idx):
    with torch.no_grad():

        px_z = vae.decoder.forward_manual_latents(mb, zs[:layer_idx], t=temperature)

        samples = vae.decoder.out_net.sample(px_z)

        ax.imshow(
            samples[sample_idx, :, :, 0] * mask + (1 - mask) * mean,
            cmap=plt.cm.gray,
            vmin=minv,
            vmax=maxv,
        )

        all_kls = np.concatenate([k[0].cpu().data.numpy().flatten() for k in kls[:layer_idx]])

        ax.set_title(f"$z_{{{layer_idx}}}$ - {(all_kls / np.log(2)).sum() / 8 / 1024:.3f}KiB")


for ax, layer_idx in zip(axes[1:], (2, 6, 12, 20, 36, len(zs) + 1)):
    plot_layer(ax, layer_idx)

for ax in axes:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
# %% codecell
############# sample images from latent space ###############

# does not work at the moment due to problems with data being on cpu/gpu
# plots should, in the best case, look like normal images
# forward_uncond is being used -> no latents are used here, everything is sampled from scratch
# tepmerature is used differently

sample_idx = 0
temperature = 0.5

fig, axes = plt.subplots(4, 6, figsize=(12, 8))

with torch.no_grad():
    for r in range(4):
        for c in range(6):
            mb = data_input.shape[0]
            px_z = vae.decoder.forward_uncond(mb, t=temperature)
            samples = vae.decoder.out_net.sample(px_z)
            axes[r, c].imshow(
                samples[sample_idx, 24:-24, :, 0] * mask[24:-24] + (1 - mask[24:-24]) * mean,
                cmap=plt.cm.gray,
                vmin=minv,
                vmax=maxv,
            )

plt.axis("off")
for ax in axes.flatten():
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
# %% codecell
all_kls = np.concatenate([k[0].cpu().data.numpy().flatten() for k in kls])
# blue
plt.hist(all_kls, log=True)
# orange
plt.hist(all_kls, bins=25, log=True)
# %% codecell
all_qms = np.concatenate([k[0].cpu().data.numpy().flatten() for k in qms[:12]])
plt.hist(all_qms, log=True, bins=25)
# %% codecell
kl_df = []
layer_bytes = []

for layer_idx, layer_kl in enumerate(kls):
    layer_df = pd.DataFrame(list(layer_kl.mean(dim=(0, 2, 3)).cpu().data.numpy()), columns=["KL"])
    layer_df["layer"] = layer_idx
    kl_df.append(layer_df)
    layer_bytes.append((layer_kl[0] / np.log(2)).sum().item() / 8)

kl_df = pd.concat(kl_df)
# %% codecell
plt.figure(figsize=(12, 4))
sns.swarmplot(x="layer", y="KL", data=kl_df, color="gray", s=2)
# %% codecell
plt.plot(layer_bytes)
plt.xlabel("Layer")
plt.ylabel("Entropy in bytes")
plt.semilogy()
# %% codecell
kl_df.groupby("layer").mean().plot()
# %% codecell
