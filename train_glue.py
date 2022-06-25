import pickle
import random
import argparse
import os
import sys
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from transformers import *
from torch.utils.data import DataLoader
from model.bert import BERT_GLUE, BERT_STS, BERT_MNLI
from task.result_calculator import *
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import scipy.stats as stats
from torch.utils.data import Dataset
from model.__main__module import *
from tqdm import tqdm
import time
from model.bert_layers import Bert_For_Att_output, Bert_For_Att_output_MLM
from model.minilm_v2 import Bert_For_minilm_v2, Bert_For_minilm_v2_MLM

from torch.nn.parallel import DistributedDataParallel as DDP


Datapath = {
    "cola" : "CoLA",
    "mnli" : "MNLI",
    "mrpc" : "MRPC",
    "rte" : "RTE",
    "sst" : "SST_2",
    "sts" : "STS_B",
    "qqp" : "QQP",
    "qnli" : "QNLI"
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_linear_decay_lr(args_lr, step, warmup_step, max_step):
    if step <= warmup_step:
        lr = args_lr * (step / warmup_step)
    else:
        #lr = args_lr * (1-step/max_step)
        lr = args_lr / (max_step - warmup_step) * (max_step - step)

    return lr

def layerwise_decay(lr, optimizer, args, config):
    layer_num = config.num_hidden_layers + 2

    for num, params in enumerate(optimizer.param_groups):
        l = lr * args.layer_wise  ** (layer_num - 1 - num)
        params["lr"] = l

    return optimizer

class create_dataset(Dataset):
    def __init__(self, mode, tokenizer, dataset):
        with open("./task/GLUE/" + Datapath[dataset.lower()] + "/dataset/"+ str(mode), 'rb') as f:
            self.datas = pickle.load(f)

        self.tokenizer = tokenizer

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def __getitem__(self, idx):
        datapoint = self.datas[idx]
        return datapoint

def padded_sequence_solo(samples):
    
    encoded_dump = []
    batch_label = []
    LM_max = 0
    for sample in samples:
        #CoLA, SST
        encoded_txt = tokenizer.encode(sample['pure_txt'])
        # print(sample['pure_txt'].lower())
        encoded_dump.append(encoded_txt)
        batch_label.append(sample['label'])
        if len(encoded_txt) > LM_max:
            LM_max = len(encoded_txt)

    if LM_max >= 128:
        LM_max = 128

    encoded_batch_txt = []
    for i, encoded_example in enumerate(encoded_dump):
        if len(encoded_example) <= LM_max:
            encoded_batch_txt.append(encoded_example + [tokenizer.pad_token_id] * (LM_max-len(encoded_example)))
        else:
            encoded_batch_txt.append(encoded_example[:LM_max])
    
    return encoded_batch_txt, batch_label

def padded_sequence_pair(samples):
    
    encoded_dump = []
    batch_label = []
    LM_max = 0
    for sample in samples:
        #MRPC, RTE, STS
        encoded_txt = tokenizer.encode(sample['sen1'] + tokenizer.sep_token + sample['sen2'])

        encoded_dump.append(encoded_txt)
        batch_label.append(sample['label'])
        if len(encoded_txt) > LM_max:
            LM_max = len(encoded_txt)

    if LM_max >= 128:
        LM_max = 128

    encoded_batch_txt = []
    for i, encoded_example in enumerate(encoded_dump):
        if len(encoded_example) <= LM_max:
            encoded_batch_txt.append(encoded_example + [tokenizer.pad_token_id] * (LM_max-len(encoded_example)))
        else:
            encoded_batch_txt.append(encoded_example[:LM_max])

    return encoded_batch_txt, batch_label

def get_padded_sequence(dataset):
    if dataset.lower() in ["cola", "sst"]:
        return padded_sequence_solo
    else:
        return padded_sequence_pair

def get_spearman(val_dataloader, model, softmax):
    model.eval()
    val_pred = []
    val_truth = []
    spearmanr_vale = []

    for batch_idx, batch in enumerate(val_dataloader):
        encoded_batch_txt, batch_label = batch
        val_batch = torch.LongTensor(encoded_batch_txt)
        val_label = batch_label
        #outputs = model(val_batch.to(device))
        attention_mask = (val_batch != tokenizer.pad_token_id)
        outputs = model(val_batch.to(device), attention_mask = attention_mask.to(device))

        p = outputs.squeeze(-1).detach().cpu().numpy()

        spearmanr_vale.append((spearmanr(p, val_label)[0]))
        val_prob = softmax(outputs)
        y_pred = np.argsort(val_prob.detach().cpu().numpy(), axis=1)
        for i in range(y_pred.shape[0]):
            val_pred.append(y_pred[i])
            val_truth.append(val_label[i])
    precision_score, recall_score, F1_score = pr_score(val_truth, val_pred)
    accuracy = top1_acc(val_truth, val_pred)

    return sum(spearmanr_vale)/len(spearmanr_vale)

#MRPC, SST, RTE
def get_accuracy(val_dataloader, model, softmax):
    model.eval()
    val_pred = []
    val_truth = []
    for batch_idx, batch in enumerate(val_dataloader):
        encoded_batch_txt, batch_label = batch
        val_batch = torch.LongTensor(encoded_batch_txt)
        val_label = torch.LongTensor(batch_label)
        attention_mask = (val_batch != tokenizer.pad_token_id)
        outputs = model(val_batch.to(device), attention_mask = attention_mask.to(device))

        val_prob = softmax(outputs)
        y_pred = np.argsort(val_prob.detach().cpu().numpy(), axis=1)
        for i in range(y_pred.shape[0]):
            val_pred.append(y_pred[i])
            val_truth.append(val_label[i])
    precision_score, recall_score, F1_score = pr_score(val_truth, val_pred)
    accuracy = top1_acc(val_truth, val_pred)

    return accuracy, F1_score

#CoLA
def get_cola_score(val_dataloader, model, softmax):
    model.eval()
    val_pred = []
    val_truth = []
    mathew = []
    mathew_gold = []
    for batch_idx, batch in enumerate(val_dataloader):
        encoded_batch_txt, batch_label = batch
        val_batch = torch.LongTensor(encoded_batch_txt)
        val_label = torch.LongTensor(batch_label)
        attention_mask = (val_batch != tokenizer.pad_token_id)
        outputs = model(val_batch.to(device), attention_mask = attention_mask.to(device))

        val_prob = softmax(outputs)
        y_pred = np.argsort(val_prob.detach().cpu().numpy(), axis=1)
        for i in range(y_pred.shape[0]):
            if int(y_pred[i][-1]) == 1:
                mathew.append(1)
            else:
                mathew.append(-1)
            if val_label[i] == 1:
                mathew_gold.append(1)
            else:
                mathew_gold.append(-1)
            val_pred.append(y_pred[i])
            val_truth.append(val_label[i])

    MCC = matthews_corrcoef(mathew_gold, mathew)
    precision_score, recall_score, F1_score = pr_score(val_truth, val_pred)
    accuracy = top1_acc(val_truth, val_pred)

    return MCC

def get_opt_parameters(model, args, config):
    if args.layer_wise > 0:
        layer_num = config.num_hidden_layers + 2
        optimizer_grouped_parameters = [{"params" : [], "lr" : args.lr * (args.layer_wise) ** (layer_num - depth), "weight_decay" : 0.0 } for depth in range(1, layer_num + 1)]

        for name, params in model.named_parameters():
            if "embeddings" in name or "embeddings_project" in name:
                optimizer_grouped_parameters[0]["params"].append(params)
            #last layer
            elif "pooler" in name or "classifier" in name:
                optimizer_grouped_parameters[-1]["params"].append(params)
            #other layer
            elif "layer" in name:
                num = int(name.split(".")[3]) + 1
                optimizer_grouped_parameters[num]["params"].append(params)
            else:
                print(name)
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    return optimizer_grouped_parameters


def train_glue(model, dataset, device, num, config, args):

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if dataset == "sts":
        criteria = nn.MSELoss()
        args.num_labels = 1
    elif dataset.lower() == "mnli":
        criteria = nn.CrossEntropyLoss()
        args.num_labels = 3
    else:
        criteria = nn.CrossEntropyLoss()

    optimizer_grouped_parameters = get_opt_parameters(model, args, config)

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-6, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    softmax = nn.Softmax(-1)

    step = 1
    iters = 1

    padded_sequence = get_padded_sequence(dataset)
    best_score = 0

    train_dataset = create_dataset('train', tokenizer, dataset)
    val_dataset = create_dataset('val', tokenizer, dataset)

    total_step = args.epochs * (len(train_dataset) // args.step_batch_size)
    warmup_step = int(0.1 * total_step)

    st = time.time()
    for epoch in tqdm(range(args.epochs)):
        Loss = 0
        Loss_len = 0
        train_dataloader = DataLoader(train_dataset, batch_size = args.step_batch_size, shuffle = True, collate_fn = padded_sequence, drop_last= True, num_workers= num)
        model.train()
        for batch in train_dataloader:
            encoded_batch_txt, batch_label = batch
            train_batch = torch.LongTensor(encoded_batch_txt)
            if dataset == "sts":
                train_label = torch.FloatTensor(batch_label).unsqueeze(1).to(device)

                attention_mask = (train_batch != tokenizer.pad_token_id)
                outputs = model(train_batch.to(device), attention_mask = attention_mask.to(device))
            else:
                train_label = torch.LongTensor(batch_label)
                with torch.cuda.amp.autocast():
                    attention_mask = (train_batch != tokenizer.pad_token_id)
                    outputs = model(train_batch.to(device), attention_mask = attention_mask.to(device))

            loss = criteria(outputs, train_label.to(device))
            Loss += loss.item()
            scaler.scale(loss).backward()
            if args.layer_wise > 0:
                lr = get_linear_decay_lr(args.lr, step, warmup_step=warmup_step, max_step=total_step)
                optimizer = layerwise_decay(lr, optimizer, args, config)
            else:
                optimizer, lr = lr_scheduler(args.lr, optimizer, step, warmup_step=warmup_step, max_step=total_step)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            iters += 1
            Loss_len += 1

        # validation
        val_dataloader = DataLoader(val_dataset, batch_size=args.step_batch_size, shuffle=True, collate_fn=padded_sequence,
                                    drop_last=False, num_workers = num)

        if dataset.lower() == "cola":
            score = get_cola_score(val_dataloader, model, softmax)
        elif dataset.lower() == "sts":
            score = get_spearman(val_dataloader, model, softmax)

        elif dataset.lower() == "mrpc":
            s, f = get_accuracy(val_dataloader, model, softmax)
            score = f
        else:
            score, _ = get_accuracy(val_dataloader, model, softmax)
        
        if score > best_score:
            best_score = score

    print(f"The best score in {dataset} | lr : {args.lr} | batch : {args.step_batch_size} | seed : {args.random_seed} | warmup : {warmup_step} | total : {total_step} | time: {round(time.time() - st, 2)} ==> {best_score}")
    return best_score

def get_model(args, configuration):
    #if args.config.lower() == "half":

    if args.from_pretrained is not None:
        #base_model = Bert_For_Att_output.from_pretrained(args.from_pretrained)
        base_model = BertModel.from_pretrained(args.from_pretrained)
        #base_model = base_model.bert
    else:
        if "ext" in args.config.lower():
            base_model = Bert_For_Att_output(configuration, True, configuration.hidden_size // 2)
            base_model = Tutor_KD(base_model, configuration)
        else:
            base_model = Bert_For_Att_output(configuration, True, None)
            base_model = Tutor_KD(base_model, configuration)

        torch.cuda.set_device(device)
        base_model.load_state_dict(torch.load(args.model_save_path))
        base_model = base_model.base_model


    if dataset == "sts":
        model = BERT_STS(base_model, tokenizer, configuration)
    elif dataset.lower() == "mnli":
        model = BERT_MNLI(base_model, tokenizer, configuration)
    else:
        model = BERT_GLUE(base_model, tokenizer, configuration)

    model.to(device)

    return model

def get_config(args):

    if args.from_pretrained is not None:
        configuration = BertConfig.from_pretrained(args.from_pretrained)
        return configuration

    if args.config == "half":
        configuration = Bert_6_layer
    elif args.config == "ext-12":
        configuration = Bert_Small_Head_12_Config
    elif args.config == "ext-6":
        configuration = Bert_Small_Head_6_Config
    elif args.config == "ext-2":
        configuration = Bert_Small_Head_2_Config

    return configuration

if __name__ == "__main__":
    import torch
    from transformers import BertForMaskedLM, ElectraModel
    import numpy as np
    from config import *
    from model.bert import Small_Bert
    from model.tinybert import TinyBertForSequenceClassification, TinyBertForPreTraining

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", default='0', help="choose gpu number: 0, 1, 2, 3", type=int)
    parser.add_argument("--data", default = "sts", help = "choose glue data : cola, rte, mrpc, sst, sts, qnli...", type = str)
    parser.add_argument("--epochs", default =5, type = int)
    parser.add_argument("--from_pretrained", default= "bert-base-uncased", help="if you load the model from pre-trained use this", type=str)
    parser.add_argument("--model_save_path", default="", help="choose model save path", type=str)
    parser.add_argument("--config", default='bert-base-uncased', help="choose model architecture from: half, extreme-12, ext-6, ext-2 ", type=str)
    parser.add_argument("--weight_decay", default=0.0, help="insert weight decay", type=float)
    parser.add_argument("--layer_wise", default= 0.0, help = "layer wise dacay", type = float)
    parser.add_argument("--batch_size", default=32, help="insert batch size", type=int)
    parser.add_argument("--step_batch_size", default=32, help="insert step batch size", type=int)
    parser.add_argument("--random_seed", default=23, help="insert step batch size", type=int)
    parser.add_argument("--lr", default=5e-5, help="insert learning rate", type=float)

    args = parser.parse_args()
    configuration = get_config(args)
    dataset = args.data
    device = torch.device("cuda:%d"%args.gpu_num)
    model_save_path = args.model_save_path

    score = []
    settings = []

    model = get_model(args, configuration)
    score = train_glue(model, dataset, device, 5, configuration, args)

    print(score)
