# CS6212 Project: Parameter-Efficient Fine-Tuning for Generative Speech Recognition with Large Language Model

This is the project repository for the CS6212 project in AY23/24 Semester 2 for Group 5 of Thong Nguyen.

## Requirements 

```
torch>=2.0.0
lightning @ git+https://github.com/Lightning-AI/lightning@master
sentencepiece
tqdm  
numpy 
jsonargparse[signatures] 
bitsandbytes  
datasets  
zstandard  
numpy
torch
tqdm
more-itertools
tiktoken==0.3.3
```

---------

## Pretrained weights

Download pretrained weights at these links and put them into the `weights` folder.

```
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/alpaca.pth
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/alpaca_a.pth
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/alpaca_b.pth
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/alpaca_c.pth
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/tokenizer.model
```


---------

## Data 

Download the data at these links and put them into the `split_data` folder.

```
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/test_1.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/test_5.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/test_10.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/test_15.pt

wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/train_1.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/train_5.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/train_10.pt
wget https://storage.googleapis.com/cs6212_thongnguyen_a0262924b/train_15.pt
```

---------

## Training

### 1. Small Adapter Version

#### a. Model receiving 1 ASR caption

`python training/WL-S.py --num_sentences 1 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### b. Model receiving 5 ASR captions

`python training/WL-S.py --num_sentences 5 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### c. Model receiving 10 ASR captions

`python training/WL-S.py --num_sentences 10 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### d. Model receiving 15 ASR captions

`python training/WL-S.py --num_sentences 15 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

### 2. Medium Adapter Version

#### a. Model receiving 1 ASR caption

`python training/WL-M.py --num_sentences 1 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### b. Model receiving 5 ASR captions

`python training/WL-M.py --num_sentences 5 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### c. Model receiving 10 ASR captions

`python training/WL-M.py --num_sentences 10 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

#### d. Model receiving 15 ASR captions

`python training/WL-M.py --num_sentences 15 --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path whisper`

---------

## Testing

### 1. Small Adapter Version

#### a. Model receiving 1 ASR caption

`python Inference/WL-S.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_1.pt --save_dir ./wers_1 --root ./runs/WL_S_0.001_1`

#### b. Model receiving 5 ASR captions

`python Inference/WL-S.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_5.pt --save_dir ./wers_5 --root ./runs/WL_S_0.001_5`

#### c. Model receiving 10 ASR captions

`python Inference/WL-S.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_10.pt --save_dir ./wers_10 --root ./runs/WL_S_0.001_10`

#### d. Model receiving 15 ASR captions

`python Inference/WL-S.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_15.pt --save_dir ./wers_15 --root ./runs/WL_S_0.001_15`

### 2. Medium Adapter Version

#### a. Model receiving 1 ASR caption

`python Inference/WL-M.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_1.pt --save_dir ./wers_1 --root ./runs/WL_M_0.001_1`

#### b. Model receiving 5 ASR captions

`python Inference/WL-M.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_5.pt --save_dir ./wers_5 --root ./runs/WL_M_0.001_5`

#### c. Model receiving 10 ASR captions

`python Inference/WL-M.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_10.pt --save_dir ./wers_10 --root ./runs/WL_M_0.001_10`

#### d. Model receiving 15 ASR captions

`python Inference/WL-M.py  --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data_path ./split_data/test_15.pt --save_dir ./wers_15 --root ./runs/WL_M_0.001_15`

