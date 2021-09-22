import numpy as np
import sklearn
import skorch
import torch

from unsupervised_behaviors.constants import Behaviors


class FrameCNN(torch.nn.Module):
    def __init__(self, num_classes=len(Behaviors), num_image_channels=1, num_initial_hidden=8):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=4),
            torch.nn.BatchNorm2d(num_image_channels),
            torch.nn.Conv2d(
                num_image_channels, num_initial_hidden, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.BatchNorm2d(num_initial_hidden),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                num_initial_hidden, num_initial_hidden * 2 ** 1, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.BatchNorm2d(num_initial_hidden * 2 ** 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                num_initial_hidden * 2 ** 1,
                num_initial_hidden * 2 ** 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(num_initial_hidden * 2 ** 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                num_initial_hidden * 2 ** 2,
                num_initial_hidden * 2 ** 3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(num_initial_hidden * 2 ** 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(
                num_initial_hidden * 2 ** 3,
                num_initial_hidden * 2 ** 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(num_initial_hidden * 2 ** 4),
            torch.nn.ReLU(),
        )

        self.clf = torch.nn.Linear(num_initial_hidden * 2 ** 4, num_classes)

    def forward(self, X, **kwargs):
        X = self.cnn(X)
        X = X.mean(dim=(2, 3))
        pred = self.clf(X)
        return pred


def FrameCNNBaseline(labels, device, cnn_kwargs={}):
    class_weights = torch.from_numpy(
        sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
    ).float()

    def roc_auc(net, X, y_true):
        y_pred = net.predict_proba(X)
        return sklearn.metrics.roc_auc_score(y_true, y_pred, multi_class="ovo")

    net = skorch.NeuralNetClassifier(
        lambda: FrameCNN(**cnn_kwargs),
        optimizer=torch.optim.AdamW,
        lr=0.001,
        max_epochs=100,
        device=device,
        batch_size=128,
        criterion=lambda: torch.nn.CrossEntropyLoss(weight=class_weights),
        predict_nonlinearity=lambda p: torch.nn.functional.softmax(p, dim=-1),
        callbacks=[
            skorch.callbacks.EpochScoring(
                roc_auc, use_caching=False, lower_is_better=False, on_train=False
            ),
            skorch.callbacks.EarlyStopping(
                monitor="valid_loss",
                patience=3,
                threshold=0.0001,
                threshold_mode="rel",
                lower_is_better=True,
            ),
        ],
        train_split=skorch.dataset.CVSplit(0.1, stratified=True),
        warm_start=False,
        verbose=True,
    )

    return net
