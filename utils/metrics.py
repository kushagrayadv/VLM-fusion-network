from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor


class Metrics(object):
  def __init__(self):
    pass

  def evaluate(self, y_pred: Tensor, y_true: Tensor) -> Dict[str, float]:
    """
      {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
      }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    # all classes
    y_pred = np.argmax(y_pred, axis=1)
    acc_3 = accuracy_score(y_true, y_pred)
    f1_score_3 = f1_score(y_true, y_pred, average="weighted")

    return {
      'accuracy': acc_3,
      'f1_score': f1_score_3
    }
