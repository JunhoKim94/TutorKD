import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import random
import argparse
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.getcwd())
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig

#from model.bert_layers import BertModel, BertForMaskedLM
from model.__main__module import *
from datetime import datetime
import torch
from dataloader import *
# from parallel
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from model.bert_layers import Bert_For_Att_output, Bert_For_Att_output_MLM
from config import *
from sampling import *

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()

parser.add_argument("--data_parallel", default='False', help="use data parallel", type=str)
parser.add_argument("--gpu_num", default='0', help="choose gpu number: 1, 2, 3", type=int)
parser.add_argument("--config", default='half', help="choose model architecture from: half, extreme-12, ext-6, ext-2 ", type=str)
parser.add_argument("--model_save_path", default = "./save_model/", help = "choose the model where you want to save", type = str)
parser.add_argument("--load_save_path", default = None, help = "load the model", type = str)
parser.add_argument("--load_policy_save_path", default = None, help = "load the policy", type = str)

parser.add_argument("--data_path", default = "./data/", type = str)

parser.add_argument("--lr", default=5e-4, help="insert learning rate", type=float)
parser.add_argument("--weight_decay", default=0.01, help="insert weight decay", type=float)
parser.add_argument("--epochs", default=1000, help="insert epochs", type=int)
parser.add_argument("--batch_size", default=128, help="insert batch size", type=int)
parser.add_argument("--step_batch_size", default=128, help="insert step batch size", type=int)
parser.add_argument("--random_seed", default=16, help="insert step batch size", type=int)
parser.add_argument("--test", default = 0, help = "test mode", type=int)
args = parser.parse_args()

if args.test == 1:
    summary = SummaryWriter(comment = 'runs/Distillation_%s_%s'%(str(args.pretrained), str(args.random_seed)))

device = torch.device("cuda:%d"%args.gpu_num)
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_config(args):
    if args.config == "half":
        configuration = Bert_6_layer
    elif args.config == "ext-12":
        configuration = Bert_Small_Head_12_Config
    elif args.config == "ext-6":
        configuration = Bert_Small_Head_6_Config
    elif args.config == "ext-2":
        configuration = Bert_Small_Head_2_Config
    
    return configuration

def get_model(args, configuration):
    if args.config.lower() == "half":
        base_model = Bert_For_Att_output(configuration, True, None)
    else:
        base_model = Bert_For_Att_output(configuration, True, configuration.hidden_size // 2)
    model = Tutor_KD(base_model, configuration).cuda()
    prediction_model = Bert_For_Att_output_MLM.from_pretrained("bert-base-uncased").to(device)

    return model, prediction_model

configuration = get_config(args)
model, prediction_model = get_model(args, configuration)

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

print(args.random_seed)
scaler = torch.cuda.amp.GradScaler()

step = 1
iters = 1

pred_config = BertConfig.from_pretrained("bert-base-uncased")

policy_layer = Last_layer(pred_config, pred_config.vocab_size)
policy_layer.to(device)

if args.load_save_path is not None:
    model_save_path = args.load_save_path
    policy_save_path = args.load_policy_save_path

    model.load_state_dict(torch.load(model_save_path, map_location = device))
    policy_layer.load_state_dict(torch.load(policy_save_path, map_location = device))


param_optimizer = list(model.named_parameters())
policy_layer_param = list(policy_layer.named_parameters())
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in policy_layer_param if not any(nd in n for nd in no_decay)],
    'weight_decay': args.weight_decay},
    {'params': [p for n, p in policy_layer_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-6, lr=args.lr)
optimizer.zero_grad()

norm = transforms.Normalize(mean=torch.zeros(args.batch_size), std = torch.ones(args.batch_size))
prediction_model.requires_grad_(False)
softmax = nn.Softmax(-1)

criterion = nn.CrossEntropyLoss()

lambda_1 = 0.5
lambda_2 = 25
lambda_3 = 1
lambda_4 = 1

path = args.data_path

temp = []
for epoch in range(args.epochs):
    Loss = 0
    Distill_Loss = 0
    PP_Loss = 0
    Policy_Loss = 0
    P_loss = 0
    Loss_len = 0
    G_loss = 0
    D_loss = 0

    Distill_D = 0

    N_ones = 0
    N_correct = 0

    print("now %s epoch..." % str(epoch + 1))
    for file in os.listdir(path):
        train_dataset = create_dataset(str(file), tokenizer, path)
        train_dataloader = DataLoader(train_dataset, batch_size=args.step_batch_size, shuffle=True,
                                      collate_fn=padded_sequence, drop_last=True, num_workers=10)

        for batch in tqdm(train_dataloader, ncols = 100):
            
            lm_embed, lm_label_embed, label_mask_, label_position = batch

            for i in range(int(args.step_batch_size / args.batch_size)):

                sub_lm_embed = lm_embed[i * args.batch_size:(i + 1) * args.batch_size]
                sub_lm_label = lm_label_embed[i * args.batch_size:(i + 1) * args.batch_size]
                sub_label_mask = label_mask_[i * args.batch_size:(i + 1) * args.batch_size]
                sub_label_position = label_position[i * args.batch_size:(i + 1) * args.batch_size]

                sub_batch = torch.LongTensor(sub_lm_embed).cuda()
                sub_lm_label = torch.LongTensor(sub_lm_label).cuda()
                sub_label_mask = torch.BoolTensor(sub_label_mask)

                attention_mask_ = (sub_batch == tokenizer.pad_token_id)
                zero_pad = torch.zeros(attention_mask_.size()).cuda()
                _attention_mask_ = zero_pad.masked_fill(attention_mask_, 1)

                with torch.cuda.amp.autocast():
            
                    outputs = prediction_model(sub_batch, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
                    t_logit = outputs[0]

                    t_hidden, t_att, t_value = outputs[-3:]
                    t_logit = t_logit[sub_label_mask == False]

                    t_hidden_input = t_hidden[-1]
                    t_hidden_input = t_hidden_input[sub_label_mask == False]

                    policy_logit = policy_layer(t_hidden_input)

                    t_logit = softmax(t_logit)
                    
                    prior_tensor, sub_batch, sub_label_position, policy_pred, num_correct, num_ones, replaced_tokens = get_policy_sample(t_logit, 
                                                                                                                    policy_logit,
                                                                                                                    sub_batch, 
                                                                                                                    sub_label_position, 
                                                                                                                    sub_lm_label, 
                                                                                                                    sub_label_mask
                                                                                                                    )

                    outputs = prediction_model(sub_batch, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
                    t_hidden, t_att, t_value = outputs[-3:]

                    t_logit_forward = outputs[0]
                    t_logit_forward = t_logit_forward[sub_label_mask == False]
                    t_logit_forward = softmax(t_logit_forward)

                    index_tensor = torch.arange(0, int(len(t_logit_forward)), dtype=int)
                    origin = t_logit_forward[index_tensor, sub_lm_label[sub_label_mask == False]]
                    rep = t_logit_forward[index_tensor, replaced_tokens]

                    g_loss = origin - rep

                    distil_loss, pp_loss, s_out = model(sub_batch, prior_tensor.cuda(), _attention_mask_, sub_label_mask.cuda(), sub_lm_label, t_hidden, t_att)

                    s_out = torch.sigmoid(s_out)
                    d_loss = torch.abs(prior_tensor[sub_label_mask == False] - s_out[sub_label_mask == False]).detach()
                    distil_loss = distil_loss.mean()

                    kl_criterion = torch.nn.KLDivLoss(reduction = "batchmean")
                    p_loss = kl_criterion(torch.softmax(policy_logit, dim = -1).log(), t_logit)

                    policy_trg = g_loss + d_loss
                    policy_loss = torch.mean(-torch.log(policy_pred) * policy_trg)

                    pp_loss = pp_loss.mean()
                    
                    loss = (distil_loss * lambda_1 + pp_loss * lambda_2 + policy_loss * lambda_3 + p_loss * lambda_4) / (args.step_batch_size / args.batch_size)
                    
                    PP_Loss += pp_loss.item() / (args.step_batch_size / args.batch_size)
                    Distill_Loss += distil_loss.item() / (args.step_batch_size / args.batch_size)
                    Policy_Loss += policy_loss.item() / (args.step_batch_size / args.batch_size)
                    G_loss += g_loss.mean().item() / (args.step_batch_size / args.batch_size)
                    D_loss += d_loss.mean().item() / (args.step_batch_size / args.batch_size)
                    P_loss += p_loss.mean().item() / (args.step_batch_size / args.batch_size)

                    N_correct += num_correct / (args.step_batch_size / args.batch_size)
                    N_ones +=  num_ones / (args.step_batch_size / args.batch_size)   

                    loss = loss.mean()
                    Loss += loss.item()
                    
                    scaler.scale(loss).backward()

            optimizer, lr = lr_scheduler(args.lr, optimizer, step, warmup_step=10000, max_step=1000000)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            iters += 1
            step += 1
            Loss_len += 1

            if iters % 500 == 0 and (args.test == 1):
                summary.add_scalar('loss/loss_a', float(Loss / Loss_len), step)
                summary.add_scalar("loss/disc_loss", float(PP_Loss/ Loss_len), step)
                summary.add_scalar("loss/distill_loss", float(Distill_Loss/ Loss_len), step)
                summary.add_scalar("loss/policy_loss", float(Policy_Loss / Loss_len), step)
                summary.add_scalar("hyp_para/lr", float(lr), step)
                summary.add_scalar("hyp_para/g_loss", float(G_loss / Loss_len), step)
                summary.add_scalar("hyp_para/d_loss", float(D_loss / Loss_len), step)
                summary.add_scalar("hyp_para/p_loss", float(P_loss / Loss_len), step)

                summary.add_scalar("hyp_para/distill_d_loss", float(Distill_D / Loss_len) , step)

                summary.add_scalar("hyp_para/num_correct", float(N_correct / Loss_len), step)
                summary.add_scalar("hyp_para/num_ones", float(N_ones / Loss_len), step)

                Loss = 0
                P_loss = 0
                Loss_len = 0
                Distill_Loss = 0
                Policy_Loss = 0
                PP_Loss = 0
                G_loss = 0
                D_loss = 0
                P_loss = 0

                Distill_D = 0
                N_correct = 0
                N_ones = 0

            if iters % 10000 == 0:
                PATH = args.model_save_path + '/Sample_%s_%s_%s_lambda_%s_%s_%s_%s.pt' % (
                    str(args.config), str(step), str(args.model), str(lambda_1), str(lambda_2), str(lambda_3), str(lambda_4))

                Policy_PATH = args.model_save_path + "/Sample_%s_%s_%s_policy_%s_%s_%s_%s.pt" % (
                    str(args.config), str(step), str(args.model), str(lambda_1), str(lambda_2), str(lambda_3), str(lambda_4))

                print("save the model")
                torch.save(policy_layer.state_dict(), Policy_PATH)
                torch.save(model.state_dict(), PATH)

summary.close()