#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L vnode-core=30
#PJM -L elapse=6:00:00
#PJM -L gpu=1
#PJM -j

module load singularity-ce/4.1.3
# singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 128 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
# singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 512 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s8" --project_name "multiple_phase_20250118_inductionhead"
singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 1024 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 32 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"
singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 0 --num_task 1 --num_seq 4 --exp_name "multi3_ih_w4_c64_e0.1_p0_t1_s16" --project_name "multiple_phase_20250118_inductionhead"