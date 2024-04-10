# train
import wandb
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass, asdict
import os
import sys
sys.path.append("/workspace/induction-head")
from data import SamplingLoader, IterDataset, SamplingDataset, MultiTaskSamplingLoader, IterDatasetFortask
from model import InputEmbedder, Transformer, TransformerICL, MultiTaskInputEmbedderV1, MultiTaskInputEmbedderV3
# from config_multi import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from configs.config_multi3 import TransformerConfig, TrainDataConfig, IWLDataConfig, ICLDataConfig, ICL2DataConfig, MainConfig
from argparse import ArgumentParser
from utils import visalize_attention
import matplotlib.pyplot as plt
import numpy as np
import os


def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) / p.shape[0]
def to_gpu_dict(dic, device="cuda:0"):
    dic = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in dic.items()}
    return dic

# モデルの指定されたレイヤーの出力と勾配を保存するクラス
class TaskVector:
    def __init__(self, model, target_layer):  # 引数：モデル, 対象のレイヤー
        self.model = model
        self.layer_output = []
        self.layer_grad = []
        
        # 特徴マップを取るためのregister_forward_hookを設定
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        # 勾配を取るためのregister_forward_hookを設定
        self.grad_handle = target_layer.register_forward_hook(self.gradient)

    # self.feature_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの出力（特徴マップ）を保存する
    def feature(self, model, input, output):
         activation = output
         self.layer_output.append(activation.to("cpu").detach())

    # self.grad_handleの定義時に呼び出されるメソッド
    ## モデルの指定されたレイヤーの勾配を保存する
    ## 勾配が存在しない場合や勾配が必要ない場合は処理をスキップ
    def gradient(self, model, input, output):
        # 勾配が無いとき
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return # ここでメソッド終了

        # 勾配を取得
        def _hook(grad): 
            # gradが定義されていないが、勾配が計算されると各テンソルのgrad属性に保存されるっぽい（詳細未確認）
            self.layer_grad.append(grad.to("cpu").detach())

        # PyTorchのregister_hookメソッド（https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html）
        output.register_hook(_hook) 

    # メモリの解放を行うメソッド、フックを解除してメモリを解放する
    def release(self):
        self.feature_handle.remove()
        self.grad_handle.remove()

# モデルの指定されたレイヤーの出力と勾配を保存するクラス
class TaskVectorInjection:
    def __init__(self, model, target_layer, injection):  # 引数：モデル, 対象のレイヤー
        self.model = model
        self.target_layer = target_layer
        self.injection = injection
        
        self.feature_handle = target_layer.register_forward_hook(self.feature)
        
    
    def feature(self, model, input, output):
        output[:,-1] = self.injection[:,-1]
        return output
        


def main(config, save_dir):
    wandb.init(project="induction-head-task-vector", config=asdict(config))
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
    iwl_dataloader = torch.utils.data.DataLoader(iwl_dataset, batch_size=trainconfig.batch_size, \
        pin_memory=True, num_workers=os.cpu_count())
    
    # task_vector_loader = MultiTaskSamplingLoader(traindataconfig, Dataset)
    # task_vector_seq_generator = task_vector_loader.get_seq_for_task_vector
    # task_vector_dataset = IterDatasetFortask(task_vector_seq_generator)
    # task_vector_dataloader = torch.utils.data.DataLoader(task_vector_dataset, batch_size=trainconfig.batch_size, \
    #     pin_memory=True, num_workers=os.cpu_count())

    # icl2loader = MultiTaskSamplingLoader(icl2dataconfig, Dataset)
    # icl2_seq_generator = icl2loader.get_seq
    # icl2_dataset = IterDataset(icl2_seq_generator)
    # icl2_dataloader = torch.utils.data.DataLoader(icl2_dataset, batch_size=trainconfig.batch_size, pin_memory=True, num_workers=os.cpu_count())

    # model
    embedder = MultiTaskInputEmbedderV3(modelconfig)
    # if not modelconfig.use_standard_transformer:
    #     model = TransformerICL(embedder, modelconfig)
    # else:
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
    for (data_dict, icl_data_dict, iwl_data_dict) in zip(tqdm(train_dataloader), icl_dataloader, iwl_dataloader):
        model.train()   
        data_dict = to_gpu_dict(data_dict, device=config.device)
        icl_data_dict = to_gpu_dict(icl_data_dict, device=config.device)
        iwl_data_dict = to_gpu_dict(iwl_data_dict , device=config.device)
        # icl2_data_dict = to_gpu_dict(icl2_data_dict , device=config.device)
        
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
                
                
                test_data_dict = iwl_data_dict["task_vector"]   
                test_data_dict = to_gpu_dict(test_data_dict, device=config.device)
                # embedder.Emb
                # atten_list
                # atten_list.0
                # atten_list.1
                # mlp_list
                # mlp_list.0
                # mlp_list.1
                # classifier
                emb = TaskVector(model, model.embedder.Emb)
                atten0 = TaskVector(model, model.atten_list[0])
                atten1 = TaskVector(model, model.atten_list[1])
                mlp0 = TaskVector(model, model.mlp_list[0])
                mlp1 = TaskVector(model, model.mlp_list[1])
                classifier = TaskVector(model, model.classifier)
                logits = model(test_data_dict["examples"], test_data_dict["labels"], test_data_dict["tasks"])
                query_logit = logits[:,-1,:]
                icl_acc = cal_acc(test_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/test_icl_acc":icl_acc}, step=step)
                
                target_layer = model.atten_list[0]
                injection = atten0.layer_output[0]
                taskinjection = TaskVectorInjection(model, target_layer, injection)
                logits = model(iwl_data_dict["examples"], iwl_data_dict["labels"] , iwl_data_dict["tasks"])
                query_logit = logits[:,-1,:]
                icl_acc = cal_acc(iwl_data_dict["labels"][:, -1], query_logit)
                wandb.log({"valid/task_vector_icl_acc":icl_acc}, step=step)
                # logit : [batch, layer, seq_len, num_class]
                # logits = model.injection(task_hideen, iwl_data_dict["examples"], iwl_data_dict["labels"] , iwl_data_dict["tasks"])
                # logits = model(icl2_data_dict["examples"], icl2_data_dict["labels"], icl2_data_dict["task"])
                # query_logit = logits[:,-1,:]
                # icl2_acc = cal_acc(icl2_data_dict["labels"][:, -1, -1], query_logit)
                # wandb.log({"valid/icl2_acc":icl2_acc}, step=step)
                for layer_i in range(modelconfig.num_atten_layer):
                    attn_img = visalize_attention(model, layer_i)
                    wandb.log({"attention/layer_{}".format(layer_i):[wandb.Image(attn_img)]}, step=step)
                    plt.close()
                del attn_img, iwl_acc, icl_acc
                
        print("\r step:",step+1,"/",trainconfig.optimize_step, end="")
        step+=1
        if step > trainconfig.optimize_step:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, config.exp_name+".pt"))
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
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--exp_name", type=str, default="some_exp")
    parser.add_argument("--num_seq", type=int, default=8)
    parser.add_argument("--task_ways", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--use_standard_transformer", action="store_true")
    parser.add_argument("--num_atten_layer", type=int, default=2)
    
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
    config.modelconfig.num_seq = parser.parse_args().num_seq
    
    # same task in seawuence
    config.traindataconfig.task_ways = config.traindataconfig.num_seq
    config.icldataconfig.task_ways = config.icldataconfig.num_seq
    config.iwldataconfig.task_ways = config.iwldataconfig.num_seq
    # config.icl2config.task_ways = parser.parse_args().num_seq
    config.modelconfig.task_ways = config.modelconfig.num_seq
    
    # config.modelconfig.n_ctx = (config.modelconfig.num_seq+1)*2
    config.modelconfig.n_ctx = (config.traindataconfig.num_seq+1)*2 -1
    
    config.modelconfig.num_layers = parser.parse_args().num_layer
    config.modelconfig.d_model = parser.parse_args().d_model
    
    config.modelconfig.use_standard_transformer = parser.parse_args().use_standard_transformer
    
    config.modelconfig.num_atten_layer = parser.parse_args().num_atten_layer
    
    config.device = parser.parse_args().device
    
    save_dir = parser.parse_args().save_dir
    
    
    
    main(config, save_dir)