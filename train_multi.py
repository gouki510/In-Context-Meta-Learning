# train
import wandb
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass, asdict
from data import SamplingLoader, IterDataset, SamplingDataset, MultiTaskSamplingLoader
from model import InputEmbedder, Transformer, TransformerICL, MultiInputEmbedder, MultiTransformerICL
from config_multi import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from argparse import ArgumentParser
import numpy as np
import os



def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) / p.shape[0]
def to_gpu_dict(dic, device="cuda:0"):
    dic = {k:v.to(device) for k,v in dic.items()}
    return dic


def main(config):
    wandb.init(project="induction-head-multi", config=asdict(config))
    trainconfig = config.trainconfig
    modelconfig = config.modelconfig
    traindataconfig = config.traindataconfig
    icldataconfig = config.icldataconfig
    iwldataconfig = config.iwldataconfig
    icl2dataconfig = config.icl2dataconfig
    # data
    Dataset = SamplingDataset(traindataconfig)
    
    trainloader = MultiTaskSamplingLoader(traindataconfig, Dataset)
    train_seq_generator = trainloader.get_seq
    train_dataset = IterDataset(train_seq_generator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    iclloader = MultiTaskSamplingLoader(icldataconfig, Dataset)
    icl_seq_generator = iclloader.get_seq
    icl_dataset = IterDataset(icl_seq_generator)
    icl_dataloader = torch.utils.data.DataLoader(icl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    iwlloader = MultiTaskSamplingLoader(iwldataconfig, Dataset)
    iwl_seq_generator = iwlloader.get_seq
    iwl_dataset = IterDataset(iwl_seq_generator)
    iwl_dataloader = torch.utils.data.DataLoader(iwl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    icl2loader = MultiTaskSamplingLoader(icl2dataconfig, Dataset)
    icl2_seq_generator = icl2loader.get_seq
    icl2_dataset = IterDataset(icl2_seq_generator)
    icl2_dataloader = torch.utils.data.DataLoader(icl2_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    # model
    embedder = MultiInputEmbedder(modelconfig)
    model = MultiTransformerICL(embedder, modelconfig)
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
        
        logits = model(data_dict["examples"], data_dict["labels"], data_dict["task"])
        query_logit = logits[:,-1,:]

        optimizer.zero_grad()
        
        loss = criterion(query_logit, data_dict["labels"][:,-1, -1])
        loss.backward()
        optimizer.step()
        train_acc = cal_acc(data_dict["labels"][:, -1, -1], query_logit)
        wandb.log({"train/acc":train_acc,"train/loss":loss}, step=step)
        
        model.eval()
        with torch.no_grad():
            logits = model(icl_data_dict["examples"], icl_data_dict["labels"], icl_data_dict["task"])
            query_logit = logits[:,-1,:]
            icl_acc = cal_acc(icl_data_dict["labels"][:, -1, -1], query_logit)
            wandb.log({"valid/icl_acc":icl_acc}, step=step)

            logits = model(iwl_data_dict["examples"], iwl_data_dict["labels"] , iwl_data_dict["task"])
            query_logit = logits[:,-1,:]
            iwl_acc = cal_acc(iwl_data_dict["labels"][:, -1, -1], query_logit)
            wandb.log({"valid/iwl_acc":iwl_acc}, step=step)

            logits = model(icl2_data_dict["examples"], icl2_data_dict["labels"], icl2_data_dict["task"])
            query_logit = logits[:,-1,:]
            icl2_acc = cal_acc(icl2_data_dict["labels"][:, -1, -1], query_logit)
            wandb.log({"valid/icl2_acc":icl2_acc}, step=step)
                
        print("\r step:",step+1,"/",trainconfig.optimize_step, end="")
        step+=1
        if step > trainconfig.optimize_step:
            break

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=512)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--p_bursty", type=float, default=1)
    
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
    
    config.device = parser.parse_args().device
    
    
    
    main(config)