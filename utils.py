import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Literal, List
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    average_precision_score,
    precision_recall_curve,
    auc
)


@dataclass
class BootstrapCurveData:
    curve_type: str
    y_axis: List[float]
    aucs: List[float]
    interpolated_x: np.ndarray
    pred_scores: np.ndarray
    y_true: np.ndarray
        
    def auc_ci(self):
        
        sorted_scores = np.array(self.aucs)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 95% confidence interval
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

        return confidence_lower, confidence_upper
    

def bootstrap_diagnostic_curve(y_pred: Union[np.ndarray, pd.Series], 
                               y_test: Union[np.ndarray, pd.Series],
                               curve_type=Literal["roc", "pr"], 
                               n_bootstraps=2000, 
                               random_state=42) -> BootstrapCurveData:
    
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
        
    rng = np.random.RandomState(random_state)

    y_axis = []
    auc_areas = []
    interpolated_x = np.linspace(0, 1, 100)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        
        if curve_type == "roc":
            fpr, tpr, _ = roc_curve(y_test[indices], y_pred[indices])
            interp_y = np.interp(interpolated_x, fpr, tpr)
            interp_y[0] = 0.0
            y_axis.append(interp_y)

            score = roc_auc_score(y_test[indices], y_pred[indices])
            auc_areas.append(score)

        if curve_type ==  "pr": 
            precision, recall, _ = precision_recall_curve(y_test[indices], y_pred[indices])

            interp_y = np.interp(interpolated_x, -recall + 1, precision)
            interp_y[0] = 0.0
            y_axis.append(interp_y[::-1])

            score = average_precision_score(y_test[indices], y_pred[indices])
            auc_areas.append(score)
        
    bootstrap_scores = BootstrapCurveData(curve_type,
                                          y_axis, 
                                          auc_areas, 
                                          interpolated_x,
                                          y_pred, 
                                          y_test)

    return bootstrap_scores