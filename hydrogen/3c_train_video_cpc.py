# %%
import os
import sys

import cloudpickle
import madgrad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import torch
import torchvision
from fastprogress.fastprogress import force_console_behavior

from unsupervised_behaviors.cpc.model import VideoConvCPC
from unsupervised_behaviors.data import MaskedVideoDataset

from shared.plotting import setup_matplotlib

master_bar, progress_bar = force_console_behavior()
setup_matplotlib()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# %%
devices = list(range(torch.cuda.device_count()))
device = f"cuda:{devices[0]}"
batch_size = 16 * len(devices)
num_accumulated_batches = 1

print(batch_size * num_accumulated_batches)

# %%
videos_path = "/srv/public/benwild/predictive/videos_2019_50000videos_32frames_random.h5"
model_path = "/srv/data/benwild/data/unsupervised_behaviors/random_video_cpc_20210929.pt"


# %%
data = MaskedVideoDataset(videos_path, ellipse_masked=False)
mean = data.file["mean"][()]
std = data.file["std"][()]

"""
torchvision.transforms.Lambda(lambda images: images.reshape(-1, 128, 128)),
torchvision.transforms.RandomAffine(
    degrees=(-25, 25),
    scale=(0.9, 1.1),
),
torchvision.transforms.RandomErasing(),
torchvision.transforms.Lambda(lambda images: images.reshape(-1, 65, 128, 128)),
"""

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(lambda images: torch.from_numpy(images)),
        torchvision.transforms.Lambda(lambda images: (images - mean) / std),
        torchvision.transforms.Lambda(
            lambda images: torch.nn.functional.interpolate(
                images, size=(64, 64), mode="bicubic", align_corners=False
            )
        ),
    ]
)

data = MaskedVideoDataset(videos_path, ellipse_masked=False, tag_masked=True, transform=transforms)

plt.imshow(data[0][0][0, 0] * std + mean, cmap=plt.cm.gray, interpolation="bicubic")

# %%
train_data, val_data = torch.utils.data.random_split(
    data, [int(0.9 * len(data)), int(0.1 * len(data))]
)

# %%
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
)
val_data_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
)

# %%
num_features = 64
num_timesteps = data[0][0].shape[1]

pre_convolutions = torch.nn.Sequential(
    torch.nn.Conv3d(1, num_features, padding=0, kernel_size=(1, 7, 7)),
    torch.nn.AvgPool3d((1, 2, 2)),
)
model = VideoConvCPC(
    num_channels=1,
    num_features=num_features,
    num_video_residual_blocks=6,
    num_embeddings=128,
    num_context=128,
    num_ahead=32,
    pre_convolutions=pre_convolutions,
    num_ahead_subsampling=2,
    subsample_length=num_timesteps,
    embedder_params={"num_residual_blocks_pre": 6, "num_residual_blocks": 0},
    contexter_params={"num_residual_blocks": 8, "kernel_size": 3},
).to(device)

learning_rate = 0.001
weight_decay = 1e-5

num_batches = 25000

# %%
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


def run_batch(data_generator, num_accumulated_batches=1):
    X_embs = []
    X_ctxs = []
    for i in range(num_accumulated_batches):
        X = next(data_generator)
        X_emb, X_ctx = model_parallel(X)

        if i < num_accumulated_batches - 1:
            X_emb = X_emb.detach()
            X_ctx = X_ctx.detach()
            model_parallel.zero_grad()

        X_embs.append(X_emb)
        X_ctxs.append(X_ctx)

    X_emb = torch.cat(X_embs)
    X_ctx = torch.cat(X_ctxs)

    batch_loss = model.cpc_loss(X_emb, X_ctx)

    return batch_loss


# %%
train_generator = load_batch(train_data_loader)
val_generator = load_batch(val_data_loader)

print()

for _ in range(num_batches - len(train_losses)):
    optimizer.zero_grad()

    try:
        with torch.no_grad():
            val_loss = run_batch(val_generator, num_accumulated_batches)

        train_loss = run_batch(train_generator, num_accumulated_batches)
    except RuntimeError as err:
        print(err)
        continue

    train_loss.backward()
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
torch.save((model, optimizer, train_losses, val_losses), model_path, pickle_module=cloudpickle)
model_path

# %%
model, optimizer, train_losses, val_losses = torch.load(model_path)
model_parallel = torch.nn.DataParallel(model, device_ids=devices)

# %%
torch.multiprocessing.set_sharing_strategy("file_system")

# %%
eval_videos_path = (
    "/srv/public/benwild/predictive/videos_2019_5000videos_32frames_allbehaviors_fixed.h5"
)
eval_data = MaskedVideoDataset(
    eval_videos_path, ellipse_masked=False, tag_masked=True, transform=transforms
)

# %%
from captum.attr import IntegratedGradients, NoiseTunnel, visualization

# %%
data_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)
x, y = next(iter(data_loader))
x.shape

# %%
class MeanModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, model_output_fun=None):
        super().__init__()

        self.model = model
        self.model_output_fun = model_output_fun

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)

        if self.model_output_fun is not None:
            outputs = self.model_output_fun(outputs)

        return outputs.mean(dim=tuple(range(1, outputs.ndim)))


mmodel = MeanModel(
    model.to("cpu"),
    lambda o: torch.abs(o[1]),
    # lambda o: torch.nn.functional.normalize(o[1], dim=1, p=2)
)

mmodel(x[:1])

# %%
ig = IntegratedGradients(mmodel.to("cpu"))
# nt = NoiseTunnel(ig)

# %%
with torch.no_grad():
    # attributions = nt.attribute((x[:1],), internal_batch_size=10, nt_type='smoothgrad_sq', nt_samples=1, stdevs=1.)[0]
    attributions = ig.attribute((x[:1],), internal_batch_size=1)[0]

# %%
attributions.sum(dim=(0, 1, 3, 4))

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(x[0, 0, x.shape[2] // 2], cmap=plt.cm.gray)
im = ax[1].imshow(attributions[0, 0, attributions.shape[2] // 2])

# %%
frame_idx = 18
plt.imshow(x[0, 0, frame_idx].numpy()[:, :, None], cmap=plt.cm.gray)
plt.show()

_ = visualization.visualize_image_attr(
    attributions[0, 0, frame_idx].numpy()[:, :, None],
    x[0, 0, frame_idx].numpy()[:, :, None],
    method="blended_heat_map",
    sign="absolute_value",
    show_colorbar=True,
    # outlier_perc=10,
    title="Overlayed Integrated Gradients",
)
plt.show()

# %%
model = model.to(device)

# %%
cpc_reps = []
labels = []

data_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)

with torch.no_grad():
    for X, y in progress_bar(data_loader, total=len(data_loader)):
        X_emb, X_ctx = model_parallel(X)
        clf_input = torch.nn.functional.normalize(X_ctx, dim=1, p=2)
        # clf_input = X_ctx
        # clf_input = clf_input[:, :, clf_input.shape[-1] // 2]
        clf_input = clf_input.mean(dim=-1)

        cpc_reps.append(clf_input.cpu().numpy())
        labels.append(y.cpu().numpy())

cpc_reps = np.concatenate(cpc_reps)
labels = np.concatenate(labels)

# %%
cpc_reps.shape

# %%
linear = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LogisticRegression(
        C=1e3,
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=10000,
        n_jobs=4,
    ),
)

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
score

# %%
logreg = sklearn.linear_model.LogisticRegression(
    C=1e3,
    multi_class="multinomial",
    class_weight="balanced",
    max_iter=10000,
    n_jobs=4,
)
logreg.fit(cpc_reps, labels)

# %%
from unsupervised_behaviors.constants import Behaviors

# %%
indices = np.argsort(logreg.coef_[Behaviors.DANCE.value])
coeffs = torch.from_numpy(logreg.coef_[Behaviors.DANCE.value])

coeffs
# %%
mmodel = MeanModel(
    model.to("cpu"),
    lambda o: torch.nn.functional.normalize(o[1], dim=1, p=2) * coeffs[None, :, None],
)

mmodel(x[:1])
ig = IntegratedGradients(mmodel.to("cpu"))

with torch.no_grad():
    attributions = ig.attribute((x[:1],), internal_batch_size=1)[0]

# %%
frame_idx = 5

attrib = attributions[0, 0, frame_idx].numpy()[:, :, None]
max_val = np.abs(attrib).max()

fig, ax = plt.subplots(figsize=(10, 8))

plt.imshow(x[0, 0, frame_idx].numpy()[:, :, None], cmap="gray")
plt.imshow(
    attributions[0, 0, frame_idx].numpy()[:, :, None],
    vmin=-1 * max_val,
    vmax=max_val,
    cmap="coolwarm_r",
    alpha=0.5,
)
plt.colorbar()

# %%
_ = visualization.visualize_image_attr(
    attributions[0, 0, frame_idx].numpy()[:, :, None],
    x[0, 0, frame_idx].numpy()[:, :, None],
    method="blended_heat_map",
    sign="all",
    show_colorbar=True,
    # outlier_perc=10,
    title="Overlayed Integrated Gradients",
)
plt.show()

# %%
frame_idx = 6

_ = visualization.visualize_image_attr(
    attributions[0, 0, frame_idx].numpy()[:, :, None],
    x[0, 0, frame_idx].numpy()[:, :, None],
    method="blended_heat_map",
    sign="all",
    show_colorbar=True,
    # outlier_perc=10,
    title="Overlayed Integrated Gradients",
)
plt.show()

# %%
frame_idx = 7

_ = visualization.visualize_image_attr(
    attributions[0, 0, frame_idx].numpy()[:, :, None],
    x[0, 0, frame_idx].numpy()[:, :, None],
    method="blended_heat_map",
    sign="all",
    show_colorbar=True,
    # outlier_perc=10,
    title="Overlayed Integrated Gradients",
)
plt.show()
