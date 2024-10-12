



CUDA_VISIBLE_DEVICES=0 python fever.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128


CUDA_VISIBLE_DEVICES=0 python po3.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 


CUDA_VISIBLE_DEVICES=0 python po2.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128

CUDA_VISIBLE_DEVICES=0 python sladder.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128

CUDA_VISIBLE_DEVICES=0 python llama_test.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128
