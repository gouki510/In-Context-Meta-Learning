# train
import wandb
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass, asdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print(sys.path)
from data import SamplingLoader, IterDataset, SamplingDataset
from model import InputEmbedder, Transformer, TransformerICL
from configs.config import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from utils import visalize_attention
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import os



def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) / p.shape[0]
def to_gpu_dict(dic, device="cuda:0"):
    dic = {k:v.to(device) for k,v in dic.items()}
    return dic


def main(config):
    wandb.init(project="induction-head-repoduce2", config=asdict(config))
    trainconfig = config.trainconfig
    modelconfig = config.modelconfig
    traindataconfig = config.traindataconfig
    icldataconfig = config.icldataconfig
    iwldataconfig = config.iwldataconfig
    icl2dataconfig = config.icl2dataconfig
    # data
    Dataset = SamplingDataset(traindataconfig)
    
    trainloader = SamplingLoader(traindataconfig, Dataset)
    train_seq_generator = trainloader.get_seq
    train_dataset = IterDataset(train_seq_generator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    iclloader = SamplingLoader(icldataconfig, Dataset)
    icl_seq_generator = iclloader.get_seq
    icl_dataset = IterDataset(icl_seq_generator)
    icl_dataloader = torch.utils.data.DataLoader(icl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    iwlloader = SamplingLoader(iwldataconfig, Dataset)
    iwl_seq_generator = iwlloader.get_seq
    iwl_dataset = IterDataset(iwl_seq_generator)
    iwl_dataloader = torch.utils.data.DataLoader(iwl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    icl2loader = SamplingLoader(icl2dataconfig, Dataset)
    icl2_seq_generator = icl2loader.get_seq
    icl2_dataset = IterDataset(icl2_seq_generator)
    icl2_dataloader = torch.utils.data.DataLoader(icl2_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    # model
    embedder = InputEmbedder(modelconfig)
    model = TransformerICL(embedder, modelconfig)
    model.to(config.device)

    # optimizer
    if trainconfig.optimizer == "adam":
        optimizer =  torch.optim.Adam(model.parameters(), lr=trainconfig.lr)
    elif trainconfig.optimizer == "adamw":
        optimizer =  torch.optim.AdamW(model.parameters(), lr=trainconfig.lr)
    elif trainconfig.optimizer == "sgd":
        optimizer =  torch.optim.SGD(model.parameters(), lr=trainconfig.lr)

    # loss
    criterion = nn.CrossEntropyLoss()
    step = 0
    for (data_dict, icl_data_dict, iwl_data_dict, icl2_data_dict) in zip(tqdm(train_dataloader), icl_dataloader, iwl_dataloader, icl2_dataloader):
        model.train()   
        data_dict = to_gpu_dict(data_dict, device=config.device)
        icl_data_dict = to_gpu_dict(icl_data_dict, device=config.device)
        iwl_data_dict = to_gpu_dict(iwl_data_dict , device=config.device)
        icl2_data_dict = to_gpu_dict(icl2_data_dict , device=config.device)
        
        logits = model(data_dict["examples"], data_dict["labels"])
        query_logit = logits[:,-1,:]

        optimizer.zero_grad()
        
        loss = criterion(query_logit, data_dict["labels"][:,-1],)
        loss.backward()
        optimizer.step()
        train_acc = cal_acc(data_dict["labels"][:, -1], query_logit)
        wandb.log({"train/acc":train_acc.cpu(), "train/loss": loss.cpu()}, step=step)
        
        if step % trainconfig.every_eval == 0:
            model.eval()
            with torch.no_grad():
                logits = model(icl_data_dict["examples"], icl_data_dict["labels"])
                query_logit = logits[:,-1,:]
                icl_acc = cal_acc(icl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl_acc":icl_acc.cpu()}, step=step)

                logits = model(iwl_data_dict["examples"], iwl_data_dict["labels"])
                query_logit = logits[:,-1,:]
                iwl_acc = cal_acc(iwl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/iwl_acc":iwl_acc.cpu()}, step=step)

                logits = model(icl2_data_dict["examples"], icl2_data_dict["labels"])
                query_logit = logits[:,-1,:]
                icl2_acc = cal_acc(icl2_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl2_acc":icl2_acc.cpu()}, step=step)
                
                if modelconfig.seq_model == "Attention":
                    for layer_i in range(modelconfig.num_atten_layer):
                        attn_img = visalize_attention(model, layer_i)
                        wandb.log({"attention/layer_{}".format(layer_i):[wandb.Image(attn_img)]}, step=step)
            
                    del attn_img, icl2_acc, iwl_acc, icl_acc
                
        print("\r step:",step+1,"/",trainconfig.optimize_step, end="")
        step+=1
        if step > trainconfig.optimize_step:
            break
        
        # gpu allocation free
        del data_dict, icl_data_dict, iwl_data_dict, icl2_data_dict, \
            logits, query_logit, loss, train_acc
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=512)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--p_bursty", type=float, default=1)
    parser.add_argument("--p_icl", type=float, default=0)
    parser.add_argument("--exp_name", type=str, default="some_exp")
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_atten_layer", type=int, default=2)
    parser.add_argument("--seq_model", type=str, default="Attention")
    parser.add_argument("--use_scaled_attention", action="store_true")
    
    config = MainConfig()
    
    # set args
    config.traindataconfig.ways = parser.parse_args().ways
    config.icldataconfig.ways = parser.parse_args().ways
    config.iwldataconfig.ways = parser.parse_args().ways
    config.icl2dataconfig.ways = parser.parse_args().ways
    
    config.traindataconfig.num_classes = parser.parse_args().num_classes
    config.icldataconfig.num_classes = parser.parse_args().num_classes
    config.iwldataconfig.num_classes = parser.parse_args().num_classes
    config.icl2dataconfig.num_classes = parser.parse_args().num_classes
    
    config.traindataconfig.eps = parser.parse_args().eps
    config.icldataconfig.eps = parser.parse_args().eps
    config.iwldataconfig.eps = parser.parse_args().eps
    config.icl2dataconfig.eps = parser.parse_args().eps
    
    config.traindataconfig.alpha = parser.parse_args().alpha
    config.icldataconfig.alpha = parser.parse_args().alpha
    config.iwldataconfig.alpha = parser.parse_args().alpha
    config.icl2dataconfig.alpha = parser.parse_args().alpha
    
    config.traindataconfig.p_bursty = parser.parse_args().p_bursty
    config.icldataconfig.p_bursty = parser.parse_args().p_bursty
    config.iwldataconfig.p_bursty = parser.parse_args().p_bursty
    config.icl2dataconfig.p_bursty = parser.parse_args().p_bursty
    
    config.traindataconfig.p_icl = parser.parse_args().p_icl
    config.icldataconfig.p_icl = parser.parse_args().p_icl
    config.iwldataconfig.p_icl = parser.parse_args().p_icl
    config.icl2dataconfig.p_icl = parser.parse_args().p_icl
    
    config.modelconfig.num_layers = parser.parse_args().num_layer
    
    config.modelconfig.num_atten_layer = parser.parse_args().num_atten_layer
    
    config.modelconfig.seq_model = parser.parse_args().seq_model
    
    config.modelconfig.d_model = parser.parse_args().d_model
    
    config.modelconfig.use_scaled_attention = parser.parse_args().use_scaled_attention
    
    config.device = parser.parse_args().device
    config.exp_name = parser.parse_args().exp_name
    
    
    
    main(config)