import numpy as np
import sklearn.metrics as skmetrics


def average_precision_at_k(y_true, y_score, k):
    rel_num = np.sum(y_true)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idx = np.argsort(-y_score)
    y_true = y_true[idx[:k]]
    relevant_num = np.sum(y_true)
    
    Lx = np.cumsum(y_true)
    Px = Lx * 1.0 / np.arange(1, min(len(Lx)+1, k+1), 1)
    ap = 0.0
    if relevant_num != 0:
        ap = np.sum(Px * y_true) / min(rel_num, k)
    recall = relevant_num / rel_num 
    
    return ap, recall


def p_r_f1_at_score(y_true, y_score, score):
    rel_num = np.sum(y_true)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idx = np.where(y_score>score)[0]
    y_pred = y_true[idx]
    relevant_num = np.sum(y_pred)
    
    precision = relevant_num / y_pred.shape[0]
    recall = relevant_num / rel_num
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def average_precision(y_true, y_score):
    return skmetrics.average_precision_score(y_true, y_score)
