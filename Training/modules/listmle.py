import torch
import torch.nn as nn
from .utils import batch_gather


class ListMLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels, indexes=None):
        preds = preds.squeeze(-1)
        labels = labels.squeeze(-1)

        preds, labels, mask = batch_gather(preds, labels, indexes)

        batch_size = preds.size(0)

        sorted_indices = torch.argsort(labels, dim=1, descending=True)
        sorted_preds = torch.gather(preds, dim=1, index=sorted_indices)

        sorted_mask = torch.gather(mask, dim=1, index=sorted_indices)
        loss = []

        for b in range(batch_size):
            k = sorted_mask[b].sum().item()
            if k <= 1:
                continue

            preds = sorted_preds[b, :k]
            log_cumsum_exp = torch.logcumsumexp(preds.flip(0), dim=0).flip(0)
            sample_loss = (log_cumsum_exp - preds).sum()
            loss.append(sample_loss)

        if not loss:
            return torch.tensor(0.0, requires_grad=True, device=preds.device)

        return torch.stack(loss).mean()


def test():
    preds = torch.tensor([0.5, 0.2, 0.8, 0.3, 0.6], dtype=torch.float32)
    labels = torch.tensor([1, 0, 2, 1.4, 2.6], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    loss_fn = ListMLELoss()
    loss = loss_fn(preds, labels, indexes)
    print(f"ListNet Loss: {loss.item()}")


if __name__ == "__main__":
    test()
