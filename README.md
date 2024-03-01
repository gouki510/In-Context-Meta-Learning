# Induction-head

## Environment
```bash
docker build -t minegishi/icl .
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it --privileged -v ~/induction-head:/workspace/induction-head -p 9990:9990 --shm-size=16gb --name minegishi_icl minegishi/icl
docker run --gpus=all -e NVIDIA_VISIBLE_DEVICES=all -it --privileged -v ~/induction-head:/workspace/induction-head-p 9990:9990 --shm-size=16gb --name minegishi_icl minegishi/icl

pip install -r requirements.txt
```

## Run
```bash
python train.py --ways B --num_class K --eps Epsilon --alpha Alpha --gpu GPU
```