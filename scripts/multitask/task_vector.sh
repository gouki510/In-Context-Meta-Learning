# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer classifier --exp_name classifier
# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer classifier --exp_name classifier
python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --target_layer atten0 --exp_name task_vector_atten0
python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer atten1 --exp_name task_vector_atten1
python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --target_layer mlp0 --exp_name task_vector_mlp0
python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer mlp1 --exp_name task_vector_mlp1
# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --target_layer atten0 --exp_name task_vector_atten0
# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer atten1 --exp_name task_vector_atten1
# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:0 --p_bursty 1 --num_task 3 --target_layer mlp0 --exp_name task_vector_mlp0
# python train/train_task_vector.py --ways 1 --num_class 64 --eps 0.1 --device cuda:1 --p_bursty 1 --num_task 3 --target_layer mlp1 --exp_name task_vector_mlp1
