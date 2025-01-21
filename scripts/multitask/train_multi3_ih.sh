# /bin/bash
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 0 --num_task 1
# python train/train_multi3.py --ways 2 --num_class 1024 --eps 0.1 --device cuda:1 --p_bursty 0 --num_task 1
# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 2 --exp_name "multi3_ih_w1_c64_e0.1_p0_t1_s2"
# python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w2_c64_e0.1_p0_t1_s4"
# python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s8" --project_name "multiple_phase_20250117_inductionhead"
# python train/train_multi3.py --ways 4 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250117_inductionhead"
# python train/train_multi3.py --ways 2 --num_class 32 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s8" --project_name "multiple_phase_20250117_inductionhead"
export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

# python train/train_multi3.py --ways 2 --num_class 128 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
python train/train_multi3.py --ways 2 --num_class 512 --eps 0.1 --device cuda:1 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s8" --project_name "multiple_phase_20250118_inductionhead"
python train/train_multi3.py --ways 2 --num_class 1024 --eps 0.1 --device cuda:1 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
python train/train_multi3.py --ways 2 --num_class 32 --eps 0.1 --device cuda:1 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"