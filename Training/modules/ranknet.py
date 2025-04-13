import torch
import torch.nn as nn
from .utils import batch_gather


class RankNetLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, preds, labels, indexes=None):
        preds = preds.squeeze(-1)
        labels = labels.squeeze(-1)

        preds, labels, _ = batch_gather(preds, labels, indexes)

        pred_diffs = preds.unsqueeze(2) - preds.unsqueeze(1)
        label_diffs = labels.unsqueeze(2) - labels.unsqueeze(1)

        S_ij = torch.sign(label_diffs)
        mask = S_ij != 0

        losses = torch.log1p(torch.exp(-self.sigma * S_ij * pred_diffs))
        losses = losses[mask]

        return losses.mean()


def test():
    preds = torch.tensor([0.5, 0.2, 0.8, 0.3, 0.6], dtype=torch.float32)
    labels = torch.tensor([1, 0, 2, 1.4, 2.6], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    loss_fn = RankNetLoss()
    loss = loss_fn(preds, labels, indexes)
    print(f"ListNet Loss: {loss.item()}")


if __name__ == "__main__":
    test()
