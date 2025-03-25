python train/train_multi3.py --ways 1 --num_class 32 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 2 --num_atten_layer 1 --optimize_step 400000  --use_standard_transformer --project_name "standard_transformer_multiple_phase_20250325_class"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 2 --num_atten_layer 1 --optimize_step 400000  --use_standard_transformer --project_name "standard_transformer_multiple_phase_20250325_class"
python train/train_multi3.py --ways 1 --num_class 128 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 2 --num_atten_layer 1 --optimize_step 400000  --use_standard_transformer --project_name "standard_transformer_multiple_phase_20250325_class"
python train/train_multi3.py --ways 1 --num_class 256 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --num_seq 4 --num_layer 2 --num_atten_layer 1 --optimize_step 400000  --use_standard_transformer --project_name "standard_transformer_multiple_phase_20250325_class"


