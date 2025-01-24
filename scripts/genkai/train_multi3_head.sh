
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L vnode-core=30
#PJM -L elapse=48:00:00
#PJM -L gpu=1
#PJM -j

module load singularity-ce/4.1.3
module load cuda/11.8.0
singularity exec --nv /home/pj24002027/ku40003286/induction-head/induction2.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 400000 --num_heads 1 --project_name "multiple_phase_20250118_WoMultiHead"
singularity exec --nv /home/pj24002027/ku40003286/induction-head/induction2.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 400000 --num_heads 3 --project_name "multiple_phase_20250118_WoMultiHead"
singularity exec --nv /home/pj24002027/ku40003286/induction-head/induction2.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 5 --project_name "multiple_phase_20250118_WoMultiHead"
singularity exec --nv /home/pj24002027/ku40003286/induction-head/induction2.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 6 --project_name "multiple_phase_20250118_WoMultiHead"
singularity exec --nv /home/pj24002027/ku40003286/induction-head/induction2.sif python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 7 --project_name "multiple_phase_20250118_WoMultiHead"

