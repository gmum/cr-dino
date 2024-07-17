import gin
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn

@gin.configurable
class MaskedBCE(nn.Module):
    """
    Masks unknown tasks.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, labels):
        mask = (labels != 0).bool()
        labels[labels == -1] = 0
        loss = self.bce(preds.float(), labels.float())
        loss = loss[mask]
        return loss.mean()


def masked_rocauc(labels, preds):
    labels = np.vstack(labels)
    preds = np.vstack(preds)
    vals = []
    for i in range(labels.shape[1]):
        labels1 = np.array(labels)[:, i]
        preds1 = np.array(preds)[:, i]
        mask = labels1 != 0
        if sum(mask) > 0:
            labels1 = labels1[mask]
            labels1[labels1 == -1] = 0
            preds1 = preds1[mask]
            try:
                vals.append(roc_auc_score(labels1.astype(int), preds1))
            except ValueError:
                vals.append(-1)
    if len(vals) == 0:
        vals = [0]
    return np.array(vals)
