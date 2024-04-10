from dataclasses import dataclass, asdict

@dataclass
class TransformerConfig:
  num_seq: int = 8
  num_layers: int = 2
  num_atten_layer: int = 2
  d_vocab: int = 32
  d_model: int = 128
  d_mlp: int = 128
  d_head: int = 128
  num_heads: int = 1
  n_ctx: int = int(((num_seq+1)*2 -1))
  act_type: str = "ReLU"
  use_cache: bool = False
  use_ln: bool = True
  p_dim: int = 65
  d_emb: int = 128
  num_classes:int = 512
  num_tasks: int = 3
  task_ways: int = num_seq
  seq_model: str = "Attention"  
  use_scaled_attention: bool = False
  use_standard_transofrmer: bool = False
  

@dataclass
class TrainDataConfig:
  num_classes: int = 512
  dim: int = 63
  num_labels: int = 32
  eps: float = 0.1
  alpha: float = 0
  item_ways: int = 2
  p_bursty: float = 1
  data_type: str = "bursty" # bursty, holdout, no_support, flip
  num_seq: int = 8
  num_holdout_classes: int = 10
  num_tasks: int = 3
  task_ways: int = 8
  p_icl: float = 0

  
@dataclass
class IWLDataConfig(TrainDataConfig):
  data_type: str = "no_support" # bursty, holdout, no_support, flip

@dataclass
class ICLDataConfig(TrainDataConfig):
  data_type: str = "holdout" # bursty, holdout, no_support, flip
  task_ways: int = 8

@dataclass
class ICL2DataConfig(TrainDataConfig):
  data_type: str = "flip" # bursty, holdout, no_support, flip
  
@dataclass
class TrainConfig:
  batch_size: int = 128
  optimize_step: int = int(4e5)
  lr: float = 0.01
  optimizer: str = "sgd" # adam, sgd, adamw
  every_eval: int = 200

@dataclass
class MainConfig:
  traindataconfig : TrainDataConfig = TrainDataConfig()
  icldataconfig: ICLDataConfig = ICLDataConfig()
  iwldataconfig: IWLDataConfig = IWLDataConfig()
  icl2dataconfig: ICL2DataConfig = ICL2DataConfig()
  modelconfig: TransformerConfig = TransformerConfig()
  trainconfig: TrainConfig = TrainConfig()
  device: str = "cuda:1"
  exp_name: str = "some_exp"
# define config