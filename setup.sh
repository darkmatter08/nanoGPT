# shell script to run on CoLab or any fresh environment

git clone https://github.com/darkmatter08/nanoGPT.git
# git checkout -b origin/jains-cleanroom
git checkout origin/jains-cleanroom
git switch -c jains-cleanroom

pip install --upgrade torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
# python train.py config/train_shakespeare_char.py --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char

python data/shakespeare/prepare.py
python train.py config/train_shakespeare.py
python sample.py --out_dir=out-shakespeare-word

python data/openwebtext/prepare.py
python train.py config/train_gpt2.py
python sample.py
