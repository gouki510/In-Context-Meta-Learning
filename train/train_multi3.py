# train
import wandb
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass, asdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import SamplingLoader, IterDataset, SamplingDataset, MultiTaskSamplingLoader
from model import InputEmbedder, Transformer, TransformerICL, MultiTaskInputEmbedderV1, MultiTaskInputEmbedderV3
# from config_multi import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from configs.config_multi2 import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from argparse import ArgumentParser
from utils import visalize_attention, example_label_extract_attention, metrics_for_circuit
import matplotlib.pyplot as plt
import numpy as np
import os



def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) / p.shape[0]
def to_gpu_dict(dic, device="cuda:0"):
    dic = {k:v.to(device) for k,v in dic.items()}
    return dic


def main(config, save_dir):
    wandb.init(project="multiple_phase_induction-head-20241224_causal_exp", config=asdict(config))
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
    iclloader.task_ind = trainloader.task_ind
    icl_seq_generator = iclloader.get_seq
    icl_dataset = IterDataset(icl_seq_generator)
    icl_dataloader = torch.utils.data.DataLoader(icl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    iwlloader = MultiTaskSamplingLoader(iwldataconfig, Dataset)
    iwlloader.task_ind = trainloader.task_ind
    iwl_seq_generator = iwlloader.get_seq
    iwl_dataset = IterDataset(iwl_seq_generator)
    iwl_dataloader = torch.utils.data.DataLoader(iwl_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    icl2loader = MultiTaskSamplingLoader(icl2dataconfig, Dataset)
    icl2loader.task_ind = trainloader.task_ind
    icl2_seq_generator = icl2loader.get_seq
    icl2_dataset = IterDataset(icl2_seq_generator)
    icl2_dataloader = torch.utils.data.DataLoader(icl2_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())
    
    # model
    embedder = MultiTaskInputEmbedderV3(modelconfig)
    if not modelconfig.use_standard_transformer:
        model = TransformerICL(embedder, modelconfig)
    else:
        model = Transformer(embedder, modelconfig)
    print(model)
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
        logits = model(data_dict["examples"], data_dict["labels"], data_dict["tasks"])
        query_logit = logits[:,-1,:]

        optimizer.zero_grad()
        
        loss = criterion(query_logit, data_dict["labels"][:, -1])
        loss.backward()
        optimizer.step()
        train_acc = cal_acc(data_dict["labels"][:, -1], query_logit)
        wandb.log({"train/acc":train_acc,"train/loss":loss}, step=step)
        
        if step % trainconfig.every_eval == 0:
            model.eval()
            with torch.no_grad():

                logits = model(icl_data_dict["examples"], icl_data_dict["labels"], icl_data_dict["tasks"])
                query_logit = logits[:,-1,:]
                icl_acc = cal_acc(icl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl_acc":icl_acc}, step=step)

                logits = model(iwl_data_dict["examples"], iwl_data_dict["labels"] , iwl_data_dict["tasks"])
                query_logit = logits[:,-1,:]
                iwl_acc = cal_acc(iwl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/iwl_acc":iwl_acc}, step=step)

                logits = model(icl2_data_dict["examples"], icl2_data_dict["labels"], icl2_data_dict["tasks"])
                query_logit = logits[:,-1,:]
                icl2_acc = cal_acc(icl2_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl2_acc":icl2_acc}, step=step)
                
                for layer_i in range(modelconfig.num_atten_layer):
                    attn_img = visalize_attention(model, layer_i)
                    wandb.log({f"attention/layer_{layer_i}":[wandb.Image(attn_img)]})
                    atten_log = example_label_extract_attention(model, layer_i, n_ctx=modelconfig.n_ctx)
                    wandb.log(atten_log)
                    metrics_log = metrics_for_circuit(model, layer_i, n_ctx=modelconfig.n_ctx)
                    wandb.log(metrics_log)
                del attn_img, iwl_acc, icl_acc
                
            os.makedirs(os.path.join(save_dir, config.exp_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, config.exp_name, config.exp_name+"_"+str(step)+".pt"))
                
        print("\r step:",step+1,"/",trainconfig.optimize_step, end="")
        step+=1
        if step > trainconfig.optimize_step:
            os.makedirs(os.path.join(save_dir, config.exp_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, config.exp_name, config.exp_name+".pt"))
            break

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument("--ways", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=512)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--p_bursty", type=float, default=1)
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--exp_name", type=str, default="some_exp")
    parser.add_argument("--num_seq", type=int, default=8)
    parser.add_argument("--task_ways", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--use_standard_transformer", action="store_true")
    parser.add_argument("--num_atten_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimize_step", type=int, default=int(4e5))
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--causal_mask_type", type=lambda x: x.split(","), default=["None", "None"])
    
    config = MainConfig()
    
    # set args
    config.traindataconfig.item_ways = parser.parse_args().ways
    config.icldataconfig.item_ways = parser.parse_args().ways
    config.iwldataconfig.item_ways = parser.parse_args().ways
    config.icl2dataconfig.item_ways = parser.parse_args().ways
    
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
    
    config.traindataconfig.num_tasks = parser.parse_args().num_tasks
    config.icldataconfig.num_tasks = parser.parse_args().num_tasks
    config.iwldataconfig.num_tasks = parser.parse_args().num_tasks
    config.icl2dataconfig.num_tasks = parser.parse_args().num_tasks
    config.modelconfig.num_tasks = parser.parse_args().num_tasks
    
    config.traindataconfig.num_seq = parser.parse_args().num_seq
    config.icldataconfig.num_seq = parser.parse_args().num_seq
    config.iwldataconfig.num_seq = parser.parse_args().num_seq
    config.icl2dataconfig.num_seq = parser.parse_args().num_seq
    config.modelconfig.num_seq = parser.parse_args().num_seq
    
    # same task in seawuence
    config.traindataconfig.task_ways = config.traindataconfig.num_seq
    config.icldataconfig.task_ways = config.icldataconfig.num_seq
    config.iwldataconfig.task_ways = config.iwldataconfig.num_seq
    config.icl2dataconfig.task_ways = config.icl2dataconfig.num_seq
    config.modelconfig.task_ways = config.modelconfig.num_seq
    
    config.modelconfig.num_heads = parser.parse_args().num_heads
    
    # config.modelconfig.n_ctx = (config.modelconfig.num_seq+1)*2
    config.modelconfig.n_ctx = (config.traindataconfig.num_seq+1)*2 -1
    
    config.modelconfig.num_layers = parser.parse_args().num_layer
    config.modelconfig.d_model = parser.parse_args().d_model
    
    config.modelconfig.use_standard_transformer = parser.parse_args().use_standard_transformer
    
    config.modelconfig.num_atten_layer = parser.parse_args().num_atten_layer
    
    config.device = parser.parse_args().device
    config.trainconfig.lr = parser.parse_args().lr
    config.trainconfig.optimize_step = parser.parse_args().optimize_step
    
    config.exp_name = f"{config.exp_name}_cls{config.traindataconfig.num_classes}_seq{config.traindataconfig.num_seq}_nt{config.traindataconfig.num_tasks}_nal{config.modelconfig.num_atten_layer}"
    
    save_dir = parser.parse_args().save_dir

    config.modelconfig.causal_mask_type = parser.parse_args().causal_mask_type
    
    
    main(config, save_dir)