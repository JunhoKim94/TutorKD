import torch
import torch.nn as nn
import copy
from torch.distributions import Categorical

softmax = nn.Softmax(-1)

def get_policy_sample(logits, policy_logits, sub_batch, sub_label_position, sub_lm_label, sub_label_mask):

    batch_size = logits.shape[0]
    prior_tensor = torch.ones(sub_batch.size(), dtype=torch.float32).cuda()
    index_tensor = torch.arange(0, int(len(logits)), dtype=int)

    prediction_score = logits

    pred = torch.softmax(policy_logits, dim = -1)
    sampler = Categorical(probs = pred)
    replaced_tokens = sampler.sample()

    origin = prediction_score[index_tensor, sub_lm_label[sub_label_mask == False]]
    rep = prediction_score[index_tensor, replaced_tokens]

    sub_batch[sub_label_mask == False] = replaced_tokens
    prior_tensor[sub_label_mask == False] = rep / origin

    policy_pred = pred[index_tensor, replaced_tokens]
    prior_tensor = prior_tensor.masked_fill(prior_tensor > 1, 1.0)

    num_correct = len(replaced_tokens[sub_lm_label[sub_label_mask == False] == replaced_tokens]) / len(replaced_tokens)

    p = prior_tensor[sub_label_mask == False]

    num_ones = len(p[p == 1]) / len(p)

    return prior_tensor, sub_batch, sub_label_position, policy_pred, num_correct, num_ones, replaced_tokens