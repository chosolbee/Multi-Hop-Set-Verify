import torch
import torch.nn as nn
from .utils import batch_gather


class LambdaRankLoss(nn.Module):
    def __init__(self, sigma=1.0, eps=1e-12):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def _dcg_gain(self, y):
        return torch.pow(2.0, y) - 1.0

    def _rank_discount(self, rank):
        return 1.0 / torch.log2(rank + 2.0)

    def forward(self, preds, labels, indexes=None):
        preds = preds.squeeze(-1)
        labels = labels.squeeze(-1)

        preds, labels, mask = batch_gather(preds, labels, indexes)

        batch_size, list_size = preds.shape

        _, true_ranks = torch.sort(labels, descending=True, dim=1)
        ranks = torch.argsort(torch.argsort(-preds, dim=1), dim=1).float()

        gain = self._dcg_gain(labels)
        discount = self._rank_discount(ranks)

        position_indices = torch.arange(list_size, device=preds.device).float()
        ideal_discount = self._rank_discount(position_indices)

        ideal_discount = ideal_discount.unsqueeze(0).expand(batch_size, -1)

        pred_diff = preds.unsqueeze(2) - preds.unsqueeze(1)
        true_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
        S_ij = torch.sign(true_diff)

        gain_diff = gain.unsqueeze(2) - gain.unsqueeze(1)
        discount_diff = discount.unsqueeze(2) - discount.unsqueeze(1)
        delta_dcg = torch.abs(gain_diff * discount_diff)

        ideal_gain = torch.gather(gain, dim=1, index=true_ranks)
        ideal_dcg = torch.sum(ideal_gain * ideal_discount, dim=1, keepdim=True) + self.eps

        delta_ndcg = delta_dcg / ideal_dcg.unsqueeze(2)

        lambda_ij = torch.log1p(torch.exp(-self.sigma * S_ij * pred_diff)) * delta_ndcg

        pairwise_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        diag_mask = ~torch.eye(list_size, dtype=torch.bool, device=preds.device).unsqueeze(0)
        final_mask = pairwise_mask * diag_mask

        lambda_ij = lambda_ij * final_mask

        lambda_per_doc = lambda_ij.sum(dim=-1)

        return (lambda_per_doc**2).sum() / (final_mask.sum() + self.eps)


def test():
    preds = torch.tensor([0.5, 0.2, 0.8, 0.3, 0.6], dtype=torch.float32)
    labels = torch.tensor([1, 0, 2, 1.4, 2.6], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    loss_fn = LambdaRankLoss()
    loss = loss_fn(preds, labels, indexes)
    print(f"LambdaRank Loss: {loss.item()}")


if __name__ == "__main__":
    test()
