python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 \
    --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 \
    --causal_mask_type None,None

# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 \
#     --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 \
#     --causal_mask_type bigram,bigram

# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 \
#     --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 \
#     --causal_mask_type label_attention,bigram

# python train/train_multi3.py --ways 1 --num_class 64 --eps 0.1 --device cuda:2 \
#     --p_bursty 1 --num_seq 4 --num_layer 1  --num_atten_layer 2 --optimize_step 1000000 \
#     --causal_mask_type chunk_example,label_attention