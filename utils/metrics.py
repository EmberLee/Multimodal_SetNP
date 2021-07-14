import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_fscore_support
import pdb

def auroc(label, pred):
    return roc_auc_score(label, pred)

def auprc(label, pred):
    return average_precision_score(label, pred)

def pr_rc_f1(label, pred):
    return precision_recall_fscore_support(label, pred, average='binary', zero_division=0) # pr, rc, f1, support: # of onset cases

def get_best_f1(label, pred):
    res = np.zeros((100, 4))
    bests = []
    for i, th in enumerate(np.arange(0., 1., 0.01)):
        pr, rc, f1, _ = pr_rc_f1(label, pred > th)
        # print(f"{th}:  {pr}    {rc}    {f1} {f1_sc}")
        res[i] = np.array([th, pr, rc, f1])

    for j in range(1, 4):
        bests.append(res[np.argmax(res[:, j])]) # save best pr, best rc, best f1
    bests = np.array(bests) # shape (3, 4), col: threshold, pr, rc, f1
    return bests[-1,3], bests
