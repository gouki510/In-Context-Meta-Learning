export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"

python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 400000 --num_heads 1 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 400000 --num_heads 2 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 3 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 4 --project_name "multiple_phase_20250117_WoMultiHead"
python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 --p_bursty 1 --num_seq 4 --num_layer 1 --num_atten_layer 2 --optimize_step 400000 --num_heads 5 --project_name "multiple_phase_20250117_WoMultiHead"

