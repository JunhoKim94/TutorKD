import copy
import numpy as np


def top1_acc(test_y, pred_y, k=1):
    p_score = 0
    for i in range(len(test_y)):
        result_at_top = pred_y[i][-k]
        if result_at_top == test_y[i]:
            p_score += 1
    return float(p_score / len(test_y))

def pr_score(test_y, pred_y, k= 1):
    precision_denominator = 0
    precision_numerator = 0
    recall_numerator = 0
    recall_denominator = 0

    for i in range(len(test_y)):
        result_at_top = pred_y[i][-k]
        if result_at_top == 1:
            precision_denominator += 1
            if result_at_top == test_y[i]:
                precision_numerator += 1
        if test_y[i] == 1:
            recall_denominator += 1
            if result_at_top == 1:
                recall_numerator += 1

    recall_score = recall_numerator/ recall_denominator
    precision_score = precision_numerator/(precision_denominator+1e-5)
    F1_score = (2 * (precision_score * recall_score))/(precision_score + recall_score + 1e-6)


    return precision_score, recall_score, F1_score

def precision_score(test_y, pred_y, count, k=1):
    p_score = []
    for i in range(len(test_y)):
        result_at_topk = pred_y[i][-k]
        if result_at_topk in test_y[i]:
            count[i] += 1
        p_score.append(float(count[i]) / float(k))

    return np.mean(p_score), count


def recall_score(test_y, pred_y, count, k=1):
    r_score = []
    for i in range(len(test_y)):
        result_at_topk = pred_y[i][-k]
        if result_at_topk in test_y[i]:
            count[i] += 1
        r_score.append(float(count[i]) / float(len(test_y[i])))

    return np.mean(r_score), count
