# Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence
Arixiv link : https://arxiv.org/abs/2505.16694 

![overview](acc_gif.gif)


## Environment
```bash
docker build -t exp/icl .
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it --privileged -v ~/induction-head:/workspace/induction-head -p 9990:9990 --shm-size=16gb --name exp_icl exp/icl
docker run --gpus=all -e NVIDIA_VISIBLE_DEVICES=all -it --privileged -v ~/induction-head:/workspace/induction-head-p 9990:9990 --shm-size=16gb --name exp_icl exp/icl

pip install -r requirements.txt
```

## Run
```bash
python train.py --ways B --num_class K --eps Epsilon --alpha Alpha --gpu GPU
```

## Citation
```
@inproceedings{
minegishi2025beyond,
title={Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence},
author={Gouki Minegishi and Hiroki Furuta and Shohei Taniguchi and Yusuke Iwasawa and Yutaka Matsuo},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Xw01vF13aV}
}
```
