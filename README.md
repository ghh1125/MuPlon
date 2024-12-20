# MuPlon

You should organize them in the following format.

```
MuPlon
    ├── data
    ├── lgnn
    ├── pretrained_models
    ├── path
    ├── data_load_utils.py
    ├── cvae_models.py
    ├── cvae_pretrain_small.py
    ├── models.py
    ├── LLM_Test.py
    ├── LLM_Test_two.py
    ├── fever.py.py
    ├── po2.py.py
    ├── po3.py.py
    ├── train.sh
    ├── llama_test.py
    ├── sladder.py
    └── utils.py
```

## Environment Setup
```bash
conda create -n MuPlon python=3.9
conda activate MuPlon
pip install -r requirements.txt
```

## Model Files

- `fever.py`: the FEVER dataset 
- `po2.py`: the PO2 dataset
- `po3.py`: the PO3 dataset
- `sladder.py`: sladder dataset
- `llama_test.py`: using the LLaMA model with fever
- `LLM_Test.py` and `LLM_Test_two.py`: OLLAMA test
- `path`: save the running path of Model
- `lgnn`: local generation feature model

## Download Model

https://drive.google.com/drive/folders/1ORZ7SjvKvmKvmpzJDRb4OydiYYs0FqZ6?usp=drive_link

https://ollama.com/

https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## Reproduction

```
CUDA_VISIBLE_DEVICES=0 python fever.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128
```

```
CUDA_VISIBLE_DEVICES=0 python po3.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 
```

```
CUDA_VISIBLE_DEVICES=0 python po2.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128
```

```
CUDA_VISIBLE_DEVICES=0 python sladder.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128
```

```
CUDA_VISIBLE_DEVICES=0 python llama_test.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128
```
# Or

## Training the Model
```
bash train.sh
```
