CUDA_VISIBLE_DEVICES=0 python main_cls_al.py \
  --dataset FedISIC \
  --al_method FEAL \
  --query_model both \
  --query_ratio 0 \
  --budget 500 \
  --al_round 5 \
  --max_round 100 \
  --batch_size 32 \
  --base_lr 5e-4 \
  --kl_weight 1e-2 \
  --display_freq 20
  --seed 42, 123, 999

--dataset
Dataset to use (e.g., FedISIC)
--al_method
Active Learning strategy (e.g., FEAL)
--query_model
Which model(s) to use for querying (main, query, or both)
--query_ratio
Ratio of data queried per round (set 0 to use --budget instead)
--budget
Total annotation budget
--al_round
Number of Active Learning rounds
--max_round
Number of training (Federated Learning) rounds per AL round
--batch_size
Batch size for local training
--base_lr
Base learning rate
--kl_weight
Weight for KL divergence in evidential loss
--display_freq
Frequency (in rounds) to display evaluation metrics
