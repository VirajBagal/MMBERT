import os
import numpy as np
import pandas as pd
import random
import math
import json

import torch
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
import cv2



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def make_df(file_path):
    paths = os.listdir(file_path)
    
    df_list = []
    
    for p in paths:
        df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
        df['category'] = p.split('_')[1]
        df['mode'] = p.split('_')[2][:-4]
        df_list.append(df)
    
    return pd.concat(df_list)

def load_data(args):
    
    train_file = open(os.path.join(args.data_dir,'trainset.json'),)
    test_file = open(os.path.join(args.data_dir,'testset.json'),)
    
    train_data = json.load(train_file)
    test_data = json.load(test_file)

    traindf = pd.DataFrame(train_data) 
    traindf['mode'] = 'train'
    testdf = pd.DataFrame(test_data)
    testdf['mode'] = 'test' 

    traindf['image_name'] = traindf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    testdf['image_name'] = testdf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))

    traindf['question_type'] = traindf['question_type'].str.lower()
    testdf['question_type'] = testdf['question_type'].str.lower()



    return traindf, testdf


#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def encode_text(caption,tokenizer, args):
    part1 = [0 for _ in range(5)]
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:args.max_position_embeddings-8] + [tokenizer.sep_token_id]
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)

    
    return tokens, segment_ids, input_mask

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class VQAMed(Dataset):
    def __init__(self, df, tfm, args, mode = 'train'):
        self.df = df.values
        self.tfm = tfm
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df[idx,1]
        question = self.df[idx, 6]
        answer = self.df[idx, 3]

        if self.mode == 'eval':
            tok_ques = self.tokenizer.tokenize(question)

        img = Image.open(path)


        if self.tfm:
            img = self.tfm(img)

            
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args)


        if self.mode != 'eval':
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long)
        else:
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), tok_ques





def calculate_bleu_score(preds,targets, idx2ans):
       
    bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split(),weights=[1]) for pred,target in zip(preds,targets)])
        
    return np.mean(bleu_per_answer)



class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, 128, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(128, args.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.type_embeddings = nn.Embedding(3, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.len = args.max_position_embeddings
    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            if torch.cuda.is_available():
                position_ids = torch.arange(self.len, dtype=torch.long).cuda()
            else:
                position_ids = torch.arange(self.len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model = models.resnet152(pretrained=True)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, img):
        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
        return v_2, v_3, v_4, v_5, v_7

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h, scores
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1  
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self,args):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.fc2 = nn.Linear(args.hidden_size*4, args.hidden_size)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self,args, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                attn_output, attn_scores = self.attention[layer_num](self.norm1(hidden_states), attention_mask)
                h = self.proj[layer_num](attn_output)
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out, attn_scores

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        base_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
    def forward(self, img, input_ids, token_type_ids, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(img)
        # h = self.embed(input_ids, token_type_ids)
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        for i in range(len(h)):
            h[i][1] = v_2[i]
        for i in range(len(h)):
            h[i][2] = v_3[i]
        for i in range(len(h)):
            h[i][3] = v_4[i]
        for i in range(len(h)):
            h[i][4] = v_5[i]
        for i in range(len(h)):
            h[i][5] = v_7[i]

        hidden_states = []
        all_attn_scores = []
        for i in range(self.n_layers):
            h, attn_scores = self.blocks(h, mask, i)
            hidden_states.append(h)
            all_attn_scores.append(attn_scores)

        return torch.stack(hidden_states, 0), torch.stack(all_attn_scores, 0)


class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        self.transformer = Transformer(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
    def forward(self, img, input_ids, segment_ids, input_mask):
        h, attn_scores = self.transformer(img, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h.mean(0).mean(1)))
        logits = self.classifier(pooled_h)
        return logits, attn_scores
    
    


def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, train_df, idx2ans):

    model.train()
    train_loss = []
    
    PREDS = []
    TARGETS = []
    
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target) in bar:
        
        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            logits, _ = model(img, question_token, segment_ids, attention_mask)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            
        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)        

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))
    
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[train_df['answer_type']=='CLOSED'] == TARGETS[train_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[train_df['answer_type']=='OPEN'] == TARGETS[train_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}


    return np.mean(train_loss), acc

def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans):
    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}

    # add bleu score code
    total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    closed_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='CLOSED'],TARGETS[val_df['answer_type']=='CLOSED'],idx2ans)
    open_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='OPEN'],TARGETS[val_df['answer_type']=='OPEN'],idx2ans)

    bleu = {'total_bleu': np.round(total_bleu, 4),  'closed_bleu': np.round(closed_bleu, 4), 'open_bleu': np.round(open_bleu, 4)}

    return val_loss, PREDS, acc, bleu
    
def test(loader, model, criterion, device, scaler, args, val_df,idx2ans):

    model.eval()

    PREDS = []
    TARGETS = []

    test_loss = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            test_loss.append(loss_np)

            pred = logits.softmax(1).argmax(1).detach()
            
            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

        test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}




    return test_loss, PREDS, acc

def final_test(loader, all_models, device, args, val_df, idx2ans):

    PREDS = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            for i, model in enumerate(all_models):
                if args.mixed_precision:
                    with torch.cuda.amp.autocast(): 
                        logits, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
 
                if i == 0:
                    pred = logits.detach().cpu().numpy()/len(all_models)
                else:
                    pred += logits.detach().cpu().numpy()/len(all_models)
            
            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS