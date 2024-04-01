# python train.py --ways 2 --num_class 512 --eps 0.1 --device cuda:0
python train.py --ways 2 --num_class 256 --eps 0.1 --device cuda:2 --p_bursty 0.5
python train.py --ways 2 --num_class 128 --eps 0.1 --device cuda:2 --p_bursty 0.5
python train.py --ways 2 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.5