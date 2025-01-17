
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 --num_heads 1 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 --num_heads 2 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_heads 4 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_heads 8 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 1000000 --num_heads 16 --project_name "multiple_phase_20250117_WoMultiHead"

