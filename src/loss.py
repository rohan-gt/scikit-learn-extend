"""Loss functions"""

# Import libraries
from typing import Tuple

import numpy as np


__author__ = "Rohan George Thampi"


def focal_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.25, gamma: float = 2) -> Tuple[str, np.ndarray, bool]:
    """Computes the focal loss for binary and multi-class classification.
    
    The focal loss is a variant of the cross-entropy loss that down-weights the
    loss for well-classified examples. The focal loss is defined as:
    
    FL(p_t) = alpha * (1 - p_t)^gamma * CE(y, p_t)
    
    where p_t is the predicted probability for the positive class, y is the true label,
    alpha is a weighting factor, and gamma is a focusing parameter.
    
    Args:
        y_true: A numpy array of shape (n_samples, n_classes) or (n_samples,) containing
            the true labels.
        y_pred: A numpy array of shape (n_samples, n_classes) or (n_samples,) containing
            the predicted probabilities.
        alpha: The weighting factor for the positive class.
        gamma: The focusing parameter.
    
    Returns:
        A tuple containing the name of the loss function, the value of the loss, and a 
        Boolean value indicating whether the loss is evaluative or not.
    """

    # Compute the cross-entropy loss
    ce_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Compute the focal loss
    focal_loss = alpha * (1 - y_pred) ** gamma * ce_loss

    return "focal_loss", focal_loss, False
