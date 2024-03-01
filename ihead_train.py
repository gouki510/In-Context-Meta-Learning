# train
import wandb
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass, asdict
from ihead_data import SamplingLoader, IterDataset
from ihead_model import InputEmbedder, Transformer
from ihead_config import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig


def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) / p.shape[0]
def to_gpu_dict(dic):
    dic = {k:v.to("cuda:1") for k,v in dic.items()}
    return dic


def main(config):
    wandb.init(project="icl-induction-head", config=asdict(config))
    trainconfig = config.trainconfig
    modelconfig = config.modelconfig
    traindataconfig = config.traindataconfig
    icldataconfig = config.icldataconfig
    iwldataconfig = config.iwldataconfig
    icl2dataconfig = config.icl2dataconfig
    # data
    trainloader = SamplingLoader(traindataconfig)
    train_seq_generator = trainloader.get_seq
    train_dataset = IterDataset(train_seq_generator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=trainconfig.batch_size)

    iclloader = SamplingLoader(icldataconfig)
    icl_seq_generator = iclloader.get_seq
    icl_dataset = IterDataset(icl_seq_generator)
    icl_dataloader = torch.utils.data.DataLoader(icl_dataset, batch_size=trainconfig.batch_size)

    iwlloader = SamplingLoader(iwldataconfig)
    iwl_seq_generator = iwlloader.get_seq
    iwl_dataset = IterDataset(iwl_seq_generator)
    iwl_dataloader = torch.utils.data.DataLoader(iwl_dataset, batch_size=trainconfig.batch_size)

    icl2loader = SamplingLoader(icl2dataconfig)
    icl2_seq_generator = icl2loader.get_seq
    icl2_dataset = IterDataset(icl2_seq_generator)
    icl2_dataloader = torch.utils.data.DataLoader(icl2_dataset, batch_size=trainconfig.batch_size)

    # model
    embedder = InputEmbedder(modelconfig)
    model = Transformer(embedder, modelconfig)
    model.to("cuda:1")

    # optimizer
    optimizer =  torch.optim.SGD(model.parameters(), lr=trainconfig.lr, momentum=0.9)

    # loss
    criterion = nn.CrossEntropyLoss()
    step = 0
    for (data_dict, icl_data_dict, iwl_data_dict, icl2_data_dict) in zip(train_dataloader, icl_dataloader, iwl_dataloader, icl2_dataloader):
        model.train()   
        data_dict = to_gpu_dict(data_dict)
        icl_data_dict = to_gpu_dict(icl_data_dict)
        iwl_data_dict = to_gpu_dict(iwl_data_dict)
        icl2_data_dict = to_gpu_dict(icl2_data_dict)
        
        logits = model(data_dict["examples"], data_dict["labels"])
        query_logit = logits[:,-1,:]

        optimizer.zero_grad()
        # print(data_dict["labels"][:,-1])
        loss = criterion(query_logit, data_dict["labels"][:,-1],)
        loss.backward()
        optimizer.step()
        train_acc = cal_acc(data_dict["labels"][:, -1], query_logit)
        wandb.log({"train/acc":train_acc,"train/loss":loss}, step=step)
        with torch.no_grad():

                logits = model(icl_data_dict["examples"], icl_data_dict["labels"])
                query_logit = logits[:,-1,:]
                icl_acc = cal_acc(icl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl_acc":icl_acc}, step=step)

                logits = model(iwl_data_dict["examples"], iwl_data_dict["labels"])
                query_logit = logits[:,-1,:]
                iwl_acc = cal_acc(iwl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/iwl_acc":iwl_acc}, step=step)

                logits = model(icl2_data_dict["examples"], icl2_data_dict["labels"])
                query_logit = logits[:,-1,:]
                icl2_acc = cal_acc(icl2_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/icl2_acc":icl2_acc}, step=step)
        step+=1
        if step > trainconfig.optimize_step:
            break

if __name__ == "__main__":
    config = MainConfig()
    main(config)