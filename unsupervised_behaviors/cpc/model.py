import typing

import numpy as np
import torch
import torchtyping
from torchtyping import TensorType  # type: ignore

from unsupervised_behaviors.cpc.layers import Contexter, Embedder, ImageResidualBlock
from unsupervised_behaviors.cpc.loss import nt_xent_loss
from unsupervised_behaviors.types import batch, channels, horizontal, time, vertical

torchtyping.patch_typeguard()


class ConvCPC(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_embeddings,
        num_context,
        num_ahead,
        num_ahead_subsampling,
        subsample_length,
        embedder_params,
        contexter_params,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.num_context = num_context
        self.num_features = num_features

        self.embedder = Embedder(
            num_features,
            num_embeddings,
            **embedder_params,
        )
        self.contexter = Contexter(num_embeddings, num_context, **contexter_params)

        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_context, num_embeddings, bias=False)
                for _ in range(num_ahead // num_ahead_subsampling)
            ]
        )

        self.num_ahead = num_ahead
        self.num_ahead_subsampling = num_ahead_subsampling
        self.subsample_length = subsample_length

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.crop_pre = None
        self.crop_post = None

    def get_crops(self, device):
        if self.crop_pre is None:
            with torch.no_grad():
                initial_length = 133
                x_dummy = torch.zeros(1, self.num_features, initial_length, device=device)
                x_postemb = self.embedder(x_dummy)

                crop = (initial_length - x_postemb.shape[-1]) // 2
                self.crop_pre = crop
                self.crop_post = crop

                x_postcontext = self.contexter(x_postemb)

                crop = x_postemb.shape[-1] - x_postcontext.shape[-1]
                self.crop_pre += crop

        return self.crop_pre, self.crop_post

    def cpc_loss(
        self,
        embeddings: TensorType["batch", "channels", "time", float],
        contexts: TensorType["batch", "channels", "time", float],
        min_from_timesteps: int = 0,
    ) -> TensorType[float]:
        num_timesteps = embeddings.shape[-1]
        batch_size = len(embeddings)
        device = embeddings.device

        from_timesteps = np.random.randint(
            low=min_from_timesteps,
            high=num_timesteps - self.num_ahead_subsampling,
            size=batch_size,
        )

        contexts = contexts.transpose(1, 2)
        contexts_from = torch.stack([contexts[i, from_timesteps[i]] for i in range(batch_size)])

        embeddings_projections = torch.stack(
            list(map(lambda p: p(contexts_from), self.projections))
        )
        embeddings_projections = torch.nn.functional.normalize(embeddings_projections, p=2, dim=-1)
        embeddings_projections.shape

        ahead = torch.arange(self.num_ahead_subsampling, self.num_ahead, self.num_ahead_subsampling)
        to_timesteps = torch.from_numpy(from_timesteps)[:, None] + ahead

        valid_lengths = np.full(batch_size, num_timesteps)
        padding = max(0, to_timesteps.max() - num_timesteps + 1)
        embeddings_pad = torch.cat(
            (embeddings, torch.zeros((batch_size, self.num_embeddings, padding), device=device)),
            dim=-1,
        )

        embeddings_transposed = embeddings_pad.transpose(2, 1)
        embeddings_to = embeddings_transposed[
            torch.arange(batch_size)[:, None].repeat(1, len(ahead)).flatten(),
            to_timesteps.flatten(),
        ].reshape(batch_size, len(ahead), -1)
        embeddings_to = torch.nn.functional.normalize(embeddings_to, p=2, dim=-1)

        embeddings_projections = torch.stack(
            list(map(lambda p: p(contexts_from), self.projections[1:]))
        )
        embeddings_projections = torch.nn.functional.normalize(embeddings_projections, p=2, dim=-1)

        # assume both are l2-normalized -> cosine similarity
        predictions_to = torch.einsum("tac,btc->tab", embeddings_projections, embeddings_to)

        batch_loss = []
        for idx, current_ahead in enumerate(ahead.numpy()):
            predictions_to_ahead = predictions_to[idx]

            has_value_to = torch.from_numpy((from_timesteps + current_ahead) < valid_lengths).to(
                device, non_blocking=True
            )
            loss = nt_xent_loss(predictions_to_ahead, has_value_to)
            loss_mean = torch.mean(loss)

            batch_loss.append(loss_mean)

        aggregated_batch_loss = sum(batch_loss) / len(batch_loss)

        return aggregated_batch_loss

    def get_representations(
        self, data: np.array, batch_size: int, device: str, aggfunc=lambda x: x.mean(axis=-1)
    ):
        with torch.no_grad():
            i = 0
            reps = []

            while i < len(data):
                batch_idxs = np.arange(i, min(len(data), i + batch_size))

                X = torch.from_numpy(data[batch_idxs]).to(device, non_blocking=True)
                X = X.transpose(1, 2)

                X_emb, X_ctx = self.forward(X)
                X_ctx_agg = aggfunc(X_ctx)

                reps.append(X_ctx_agg)
                i += batch_size

        reps = torch.cat(reps).cpu().numpy()

        return reps

    def forward(
        self,
        X: TensorType["batch", "channels", "time", float],
    ):
        batch_size = len(X)
        device = X.device

        self.get_crops(device)

        X_emb = self.embedder(X)

        # TODO: only pad ctxter crop_pre
        # this works only if embedder has kernel_size=1
        X_emb_padded = torch.cat(
            (torch.zeros((batch_size, self.num_embeddings, self.crop_pre), device=device), X_emb),
            dim=-1,
        )

        X_ctx = self.contexter(X_emb_padded)

        return X_emb, X_ctx


class ImageConvCPC(ConvCPC):
    def __init__(
        self,
        num_image_channels: int,
        num_features: int,
        num_image_residual_blocks: int,
        tile_size: int,
        frn_norm: bool = True,
        aggfunc: typing.Callable[[TensorType], TensorType] = lambda x: x.mean(dim=(2, 3)),
        **kwargs,
    ):
        super().__init__(num_features=num_features, **kwargs)

        self.tile_size = tile_size
        self.aggfunc = aggfunc
        self.num_image_channels = num_image_channels
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(num_image_channels, num_features, padding=0, kernel_size=3),
            *(
                ImageResidualBlock(num_features, padding=0, kernel_size=3, frn_norm=frn_norm)
                for _ in range(num_image_residual_blocks)
            ),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        X: TensorType["batch", "channels", "vertical", "horizontal", float],
    ):
        X_tiles = torch.nn.functional.unfold(
            X, (self.tile_size, self.tile_size), stride=self.tile_size // 2
        )

        X_tiles_conv = X_tiles.transpose(2, 1)
        X_tiles_conv = X_tiles_conv.reshape(
            X_tiles_conv.shape[0] * X_tiles_conv.shape[1],
            self.num_image_channels,
            self.tile_size,
            self.tile_size,
        )
        X_tiles_conv = self.aggfunc(self.convolutions(X_tiles_conv))
        X_tiles = X_tiles_conv.reshape(X_tiles.shape[0], X_tiles.shape[-1], -1)
        X_tiles = X_tiles.transpose(2, 1)

        return super().forward(X_tiles)


class ColumnImageConvCPC(ImageConvCPC):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(
        self,
        X: TensorType["batch", "channels", "vertical", "horizontal", float],
    ):
        X_tiles = torch.nn.functional.unfold(
            X, (self.tile_size, self.tile_size), stride=self.tile_size // 2
        )

        X_tiles_conv = X_tiles.transpose(2, 1)
        X_tiles_conv = X_tiles_conv.reshape(
            X_tiles_conv.shape[0] * X_tiles_conv.shape[1],
            self.num_image_channels,
            self.tile_size,
            self.tile_size,
        )
        X_tiles_conv = self.aggfunc(self.convolutions(X_tiles_conv))
        X_tiles = X_tiles_conv.reshape(X_tiles.shape[0], X_tiles.shape[-1], -1)

        X_tiles = X_tiles.reshape(
            len(X_tiles) * self.subsample_length, self.subsample_length, X_tiles.shape[-1]
        )
        X_tiles = X_tiles.transpose(2, 1)

        return ConvCPC.forward(self, X_tiles)
