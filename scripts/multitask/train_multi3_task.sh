# python train/train_multi3.py --ways 1 --num_class 32 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 1
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 1
# python train/train_multi3.py --ways 1 --num_class 128 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 2
# python train/train_multi3.py --ways 1 --num_class 256 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 4
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 2
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 4
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 8
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 16
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 32
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 64
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 128