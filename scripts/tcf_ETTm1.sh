export CUDA_VISIBLE_DEVICES=0

model_name=tcf

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --batch_size 32 \
  --d_ff 128 \
  --d_core 64 \
  --des 'Exp' \
  --lradj cosine \
  --freq t \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --d_core 64 \
  --d_ff 128 \
  --batch_size 32 \
  --lradj cosine \
  --des 'Exp' \
  --freq t \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 32 \
  --d_core 64 \
  --lradj cosine \
  --des 'Exp' \
  --freq t \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --d_ff 512 \
  --d_core 64 \
  --batch_size 32 \
  --lradj cosine \
  --des 'Exp' \
  --freq t \
  --itr 1