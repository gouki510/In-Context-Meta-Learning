# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.8 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 300000
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.6 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 300000
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.4 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 300000
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.2 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 300000
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.0 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 300000
# python train/train_multi3.py --ways 4 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1.0 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 300000 --num_task 2
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.8 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_task 3 --exp_name 20240704
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.6 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_task 3 --exp_name 20240704
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.4 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_task 3 --exp_name 20240704
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.2 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_task 3 --exp_name 20240704
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0.0 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_task 2 --exp_name 20240704