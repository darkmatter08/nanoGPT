# shell script to run on CoLab or any fresh environment

# wandb api key: pull from https://wandb.ai/home

git clone https://github.com/darkmatter08/nanoGPT.git
cd nanoGPT
git checkout origin/jains-cleanroom
git switch -c jains-cleanroom

pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install --upgrade torch
pip install --upgrade torch numpy transformers datasets tiktoken wandb tqdm networkx
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
# python train.py config/train_shakespeare_char.py --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char

python data/shakespeare/prepare.py
python train.py config/train_shakespeare.py
python sample.py --out_dir=out-shakespeare-word

python data/openwebtext/prepare.py
python train.py config/train_gpt2.py
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py  # multi-gpu
python sample.py

# Baselines:
# python train.py eval_gpt2
# python train.py eval_gpt2_xl