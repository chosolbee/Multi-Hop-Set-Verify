import torch


def batch_gather(preds, labels, indexes=None):
    if not indexes:
        return preds.unsqueeze(0), labels.unsqueeze(0), torch.ones_like(preds).unsqueeze(0)

    unique_indexes = torch.unique(indexes)
    grouped_preds = []
    grouped_labels = []

    for idx in unique_indexes:
        mask = indexes == idx
        if mask.sum() < 2:
            continue
        grouped_preds.append(preds[mask])
        grouped_labels.append(labels[mask])

    if len(grouped_preds) == 0:
        return torch.tensor(0.0, device=preds.device)

    max_len = max(len(g) for g in grouped_preds)
    padded_preds = torch.stack([torch.cat([p, torch.zeros(max_len - len(p), device=p.device)]) for p in grouped_preds])
    padded_labels = torch.stack([torch.cat([l, torch.zeros(max_len - len(l), device=l.device)]) for l in grouped_labels])
    mask = torch.stack(
        [torch.cat([torch.ones(len(p), device=p.device), torch.zeros(max_len - len(p), device=p.device)]) for p in grouped_preds]
    ).bool()

    return padded_preds, padded_labels, mask
