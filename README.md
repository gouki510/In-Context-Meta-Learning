# Induction-head
Arixiv link : https://arxiv.org/abs/2505.16694 

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
