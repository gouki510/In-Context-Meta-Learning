#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L vnode-core=14
#PJM -L elapse=6:00:00
#PJM -L gpu=1
#PJM -j

module load singularity-ce/4.1.3
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 32 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 128 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 256 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 512 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 1024 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"
singularity exec --nv induction.sif python train/train_multi3.py --ways 1 --num_class 2048 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --project_name "multiple_phase_20250118_taskwise_acc"

