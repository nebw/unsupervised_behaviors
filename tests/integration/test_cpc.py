import numpy as np
import pytest
import torch

from unsupervised_behaviors.cpc.model import ConvCPC


class TestConvCPC:
    batch_size = 2
    device = "cpu"
    num_features = 8
    num_embeddings = 8
    num_context = 8
    num_ahead = 4
    num_ahead_subsampling = 2
    num_timesteps = 8
    embedder_params = {"num_residual_blocks_pre": 1, "num_residual_blocks": 1}
    contexter_params = {"num_residual_blocks": 1, "kernel_size": 3}

    @pytest.fixture
    def model(self):
        return ConvCPC(
            num_features=self.num_features,
            num_embeddings=self.num_embeddings,
            num_context=self.num_context,
            num_ahead=self.num_ahead,
            num_ahead_subsampling=self.num_ahead_subsampling,
            subsample_length=self.num_timesteps,
            embedder_params=self.embedder_params,
            contexter_params=self.contexter_params,
        )

    @pytest.fixture
    def data(self):
        return torch.randn(self.batch_size, self.num_features, self.num_timesteps)

    @pytest.fixture
    def outputs(self, model, data):
        return model(data)

    def test_initialize(self, model):
        assert model is not None

    def test_forward(self, outputs):
        X_emb, X_ctx = outputs

        assert torch.all(torch.isfinite(X_emb))
        assert torch.all(torch.isfinite(X_ctx))

    def test_loss(self, model, outputs):
        loss = model.cpc_loss(*outputs)

        assert torch.isfinite(loss)

    def test_get_representations(self, model, data):
        X_rep = model.get_representations(data.numpy(), self.batch_size, self.device)

        assert X_rep.shape == (self.batch_size, self.num_context)
        assert np.all(np.isfinite(X_rep))
