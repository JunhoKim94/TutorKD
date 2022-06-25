import torch.nn as nn
import torch
import copy
import os
from transformers import *
#from transformers import BertForMaskedLM
from model.act import *


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    #targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    targets_prob = targets

    loss = - targets_prob * student_likelihood
    loss = torch.sum(loss, dim = -1)

    return loss
    #return torch.mean(- targets_prob * student_likelihood, dim = -1).sum()

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
BertLayerNorm = torch.nn.LayerNorm

class Last_layer(nn.Module):
    def __init__(self, config, o_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, o_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(o_dim))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.decoder(hidden_states)

        return output

class Tutor_KD(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.pp_head = Last_layer(config, 1)
        self.mse_loss = nn.MSELoss(reduction = "none")

        self.fit_dense = nn.Linear(config.hidden_size, 768)
        self.pp_criterion = nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, input_ids, pp_label, _attention_mask_, mask_, lm_label, t_hidden, t_att):

        att_loss = 0. 
        rep_loss = 0.

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        s_hidden, s_att, s_value = outputs[-3:]

        t_layer_num = len(t_att)
        s_layer_num = len(s_att)

        assert t_layer_num % s_layer_num == 0
        
        layers_per_block = int(t_layer_num / s_layer_num)
        new_t_att = [t_att[(i + 1) * layers_per_block - 1] for i in range(s_layer_num)]
        new_t_hidden = [t_hidden[i * layers_per_block] for i in range(s_layer_num + 1)]
        
        for s, t in zip(s_att, new_t_att):
            att_loss += self.mse_loss(s,t)

        for s, t in zip(s_hidden, new_t_hidden):
            rep_loss += self.mse_loss(self.fit_dense(s),t)
            
        att_loss = torch.mean(att_loss, dim = -1).squeeze(-1)
        att_loss = torch.mean(att_loss, dim = 1).squeeze(1)
        rep_loss = torch.mean(rep_loss, dim = -1).squeeze(-1)

        loss = att_loss + rep_loss

        #Get loss      
        pp_outputs = self.pp_head(outputs[0])
        pp_loss = self.pp_criterion(pp_outputs.squeeze(-1), pp_label)

        return loss, pp_loss, pp_outputs.squeeze(-1)

class MINILM_Only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.kl_loss = nn.KLDivLoss(reduction = "batchmean")

    def forward(self, input_ids, _attention_mask_, t_hidden, t_att, t_value):

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
        s_hidden, s_att, s_value = outputs[-3:]

        t_value = t_value[-1]
        s_value = s_value[-1]
        
        t_value_dot = torch.matmul(t_value, t_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        s_value_dot = torch.matmul(s_value, s_value.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        t_value_dot = torch.softmax(t_value_dot, dim = -1)
        s_value_dot = torch.softmax(s_value_dot, dim = -1)

        s_a = torch.softmax(s_att[-1], dim = -1)
        t_a = torch.softmax(t_att[-1], dim = -1)

        att_loss = self.kl_loss(s_a.log(), t_a)
        value_loss = self.kl_loss(s_value_dot.log(), t_value_dot)

        loss = att_loss + value_loss

        loss /= input_ids.shape[1]
        loss /= self.config.num_attention_heads

        return loss


class MINILMv2_Only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_
        self.config = config

        #self.num_relation = 48
        self.num_relation = 12

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.kl_loss = nn.KLDivLoss(reduction = "batchmean")


    def forward(self, input_ids, _attention_mask_, t_hidden, t_att, t_value, t_queries, t_keys):

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        outputs = self.base_model(input_ids, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)

        s_hidden, s_att, s_value, s_queries, s_keys = outputs[-5:]

        t_queries = t_queries[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)
        t_keys = t_keys[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)
        t_value = t_value[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)

        s_queries = s_queries[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)
        s_keys = s_keys[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)
        s_value = s_value[-1].reshape(batch_size, -1).view(batch_size, self.num_relation, seq_length, -1)

        total_loss = 0
        for s_r, t_r in zip([s_queries, s_keys, s_value], [t_queries, t_keys, t_value]):
            t_dot = torch.matmul(t_r, t_r.transpose(-1, -2)) / math.sqrt(t_r.shape[-1])
            s_dot = torch.matmul(s_r, s_r.transpose(-1, -2)) / math.sqrt(s_r.shape[-1])

            t_dot = torch.softmax(t_dot, dim = -1)
            s_dot = torch.softmax(s_dot, dim = -1)

            loss = self.kl_loss(s_dot.log(), t_dot)
            loss /= input_ids.shape[1]
            loss /= self.num_relation

            total_loss += loss

        return total_loss


def lr_scheduler(args_lr, optimizer, step, warmup_step, max_step):
    if step <= warmup_step:
        lr = args_lr * (step / warmup_step)
    else:
        #lr = args_lr * (1-step/max_step)
        lr = args_lr / (max_step - warmup_step) * (max_step - step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer, lr

