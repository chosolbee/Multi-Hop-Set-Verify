import torch
import torch.nn as nn
from .utils import batch_gather


class ListNetLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, preds, labels, indexes=None):
        preds = preds.squeeze(-1)
        labels = labels.squeeze(-1)

        preds, labels, mask = batch_gather(preds, labels, indexes)
        preds.masked_fill_(~mask, float("-inf"))
        labels.masked_fill_(~mask, float("-inf"))

        p_preds = torch.softmax(preds, dim=1)
        p_labels = torch.softmax(labels, dim=1)

        loss = -torch.sum(p_labels * torch.log(p_preds + self.eps), dim=1)

        return loss.mean()


def test():
    preds = torch.tensor([0.5, 0.2, 0.8, 0.3, 0.6], dtype=torch.float32)
    labels = torch.tensor([1, 0, 2, 1.4, 2.6], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    loss_fn = ListNetLoss()
    loss = loss_fn(preds, labels, indexes)
    print(f"ListNet Loss: {loss.item()}")


if __name__ == "__main__":
    test()
