import torch
import torchtyping
from torchtyping import TensorType  # type: ignore

from unsupervised_behaviors.types import batch, channels, time

torchtyping.patch_typeguard()


class CausalConv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        downsample=False,
        **conv1d_kwargs,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.downsample = downsample
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            **conv1d_kwargs,
        )

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.conv(x)

        # remove trailing padding
        x = x[:, :, self.pad : -self.pad]

        if self.downsample:
            x = x[:, :, :: self.conv.dilation[0]]

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, padding, kernel_size, layer_norm=False, causal=False, **kwargs):
        super().__init__()

        # TODO: add dilation
        self.offset = (kernel_size - 1) // 2
        self.causal = causal

        self.conv1 = torch.nn.Conv1d(channels, channels, padding=padding, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv1d(channels, channels, padding=padding, kernel_size=kernel_size)

        self.layer_norm = layer_norm
        self.layer_norm1 = torch.nn.LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = torch.nn.LayerNorm(channels) if layer_norm else None

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x_ = x

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm1(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv1(x)

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm2(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)

        if self.offset > 0:
            if self.causal:
                x_ = x_[:, :, 4 * self.offset :]
            else:
                x_ = x_[:, :, 2 * self.offset : -2 * self.offset]
        return x_ + x


class Embedder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layer_norm=True,
        input_dropout=0,
        head_dropout=0,
        num_residual_blocks_pre=2,
        kernel_size_pre=1,
        num_residual_blocks=3,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__()

        self.input_dropout = torch.nn.Dropout2d(p=input_dropout)
        self.head_dropout = torch.nn.Dropout2d(p=head_dropout)

        self.dropout = torch.nn.Dropout(input_dropout)
        self.head = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

        self.convolutions = torch.nn.Sequential(
            *(
                ResidualBlock(
                    out_channels, kernel_size=kernel_size_pre, padding=0, layer_norm=layer_norm
                )
                for _ in range(num_residual_blocks_pre)
            ),
            *(
                ResidualBlock(
                    out_channels, kernel_size=kernel_size, padding=0, layer_norm=layer_norm
                )
                for _ in range(num_residual_blocks)
            ),
        )

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.input_dropout(x)
        x = self.head(x)
        x = self.head_dropout(x)

        x = self.convolutions(x)

        return x


class CausalResidualBlock(torch.nn.Module):
    def __init__(self, channels, layer_norm=False, **kwargs):
        super().__init__()

        self.conv1 = CausalConv1D(channels, channels, dilation=2)
        self.conv2 = CausalConv1D(channels, channels, dilation=1)

        self.layer_norm = layer_norm
        self.layer_norm1 = torch.nn.LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = torch.nn.LayerNorm(channels) if layer_norm else None

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x_ = x

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm1(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv1(x)

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm2(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)

        return x_ + x


class Contexter(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layer_norm=True,
        dropout=0,
        num_residual_blocks=8,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__()

        self.head = CausalConv1D(in_channels, out_channels)
        self.convolutions = torch.nn.Sequential(
            *(
                ResidualBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    padding=0,
                    layer_norm=layer_norm,
                    causal=True,
                )
                for i in range(num_residual_blocks)
            )
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.head(x)
        x = self.convolutions(x)
        x = self.dropout(x)

        return x


class FilterResponseNormNd(torch.nn.Module):
    def __init__(self, ndim, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], "FilterResponseNorm only supports 3d, 4d or 5d inputs."
        super().__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = torch.nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = torch.nn.Parameter(torch.Tensor(*shape))
        self.beta = torch.nn.Parameter(torch.Tensor(*shape))
        self.tau = torch.nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super().__init__(3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super().__init__(4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super().__init__(5, num_features, eps=eps, learnable_eps=learnable_eps)


class ImageResidualBlock(torch.nn.Module):
    def __init__(self, channels, padding, kernel_size, frn_norm=True, **kwargs):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(channels, channels, padding=padding, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv2d(channels, channels, padding=padding, kernel_size=kernel_size)

        if frn_norm:
            self.frn_norm_1 = FilterResponseNorm2d(channels)
            self.frn_norm_2 = FilterResponseNorm2d(channels)

        self.frn_norm = frn_norm

    def forward(self, x):
        x_ = x

        if self.frn_norm:
            x = self.frn_norm_1(x)
        else:
            x = torch.nn.functional.leaky_relu(x)
        x = self.conv1(x)

        if self.frn_norm:
            x = self.frn_norm_2(x)
        else:
            x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)

        crop = (x_.shape[-1] - x.shape[-1]) // 2

        return x_[:, :, crop:-crop, crop:-crop] + x
