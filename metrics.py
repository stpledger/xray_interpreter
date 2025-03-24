import torch

def log_confusion(pl_module, y_hat, y, split: str):
    """
    Logs overall confusion metrics.
    """
    pred = (torch.sigmoid(y_hat) > 0.5).float()
    TP = ((pred == 1) & (y == 1)).sum()
    TN = ((pred == 0) & (y == 0)).sum()
    FP = ((pred == 1) & (y == 0)).sum()
    FN = ((pred == 0) & (y == 1)).sum()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    pl_module.log(f'{split}/accuracy', accuracy)
    pl_module.log(f'{split}/precision', precision)
    pl_module.log(f'{split}/recall', recall)
    pl_module.log(f'{split}/f1_score', f1_score)

def log_class_stats(pl_module, y_hat, y, label_names, split: str):
    """
    Logs per-class metrics and average probabilities.
    """
    prob = torch.sigmoid(y_hat)
    pred = (prob > 0.5).float()
    for i, label in enumerate(label_names):
        TP = ((pred[:, i] == 1) & (y[:, i] == 1)).sum()
        TN = ((pred[:, i] == 0) & (y[:, i] == 0)).sum()
        FP = ((pred[:, i] == 1) & (y[:, i] == 0)).sum()
        FN = ((pred[:, i] == 0) & (y[:, i] == 1)).sum()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        pl_module.log(f'{label}/{split}/accuracy', accuracy)
        pl_module.log(f'{label}/{split}/precision', precision)
        pl_module.log(f'{label}/{split}/recall', recall)
        pl_module.log(f'{label}/{split}/f1_score', f1_score)

        # Compute average probabilities for true and false labels
        true_mask = y[:, i] == 1
        false_mask = y[:, i] == 0
        if true_mask.sum() > 0:
            avg_prob_true = prob[:, i][true_mask].mean()
        else:
            avg_prob_true = torch.tensor(0.0, device=prob.device)
        if false_mask.sum() > 0:
            avg_prob_false = prob[:, i][false_mask].mean()
        else:
            avg_prob_false = torch.tensor(0.0, device=prob.device)
        pl_module.log(f'{label}/{split}/avg_prob_true', avg_prob_true)
        pl_module.log(f'{label}/{split}/avg_prob_false', avg_prob_false)