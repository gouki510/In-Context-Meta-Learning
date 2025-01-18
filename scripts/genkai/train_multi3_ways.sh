#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L vnode-core=30
#PJM -L elapse=6:00:00
#PJM -L gpu=1
#PJM -j

module load singularity-ce/4.1.3
# singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --exp_name task_w1_c64_e0.1_p1_t1_s4 --project_name "multiple_phase_20250118_ways"
singularity exec --nv induction.sif python train/train_multi3.py --ways 2 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --exp_name task_w1_c64_e0.1_p1_t1_s4 --project_name "multiple_phase_20250118_ways"
singularity exec --nv induction.sif python train/train_multi3.py --ways 4 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --exp_name task_w1_c64_e0.1_p1_t2_s4 --project_name "multiple_phase_20250118_ways"
singularity exec --nv induction.sif python train/train_multi3.py --ways 8 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --exp_name task_w1_c64_e0.1_p1_t4_s4 --project_name "multiple_phase_20250118_ways"
