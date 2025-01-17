
python train/train_multi3.py --ways 1 --num_class 32 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c128_e0.1_p1_t2_s2
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c128_e0.1_p1_t2_s2
python train/train_multi3.py --ways 1 --num_class 128 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c128_e0.1_p1_t2_s2
python train/train_multi3.py --ways 1 --num_class 256 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c256_e0.1_p1_t4_s2
python train/train_multi3.py --ways 1 --num_class 512 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c512_e0.1_p1_t8_s2
python train/train_multi3.py --ways 1 --num_class 1024 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c1024_e0.1_p1_t8_s2
python train/train_multi3.py --ways 1 --num_class 2048 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c2048_e0.1_p1_t8_s2
