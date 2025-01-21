# python train/train_multi3.py --ways 1 --num_class 32 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 1
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 1
# python train/train_multi3.py --ways 1 --num_class 128 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 2
# python train/train_multi3.py --ways 1 --num_class 256 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 4
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 1 --num_seq 2 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --exp_name task_w1_c64_e0.1_p1_t1_s2
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:3 --p_bursty 1 --num_task 2 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250117_task_ramdom"
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:3 --p_bursty 1 --num_task 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250117_task_ramdom"
# python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:3 --p_bursty 0 --num_task 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250117_task_ramdom"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:3 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250117_task_ramdom"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:3 --p_bursty 1 --num_task 4 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250117_task_ramdom"
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 16
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 32
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 64
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_task 128