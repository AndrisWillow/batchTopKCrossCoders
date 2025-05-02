### Create venv

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### login to hugging face
huggingface-cli login

### login to wanb
wandb login

### Affected scripts

train-crosscoder.sh
train_crosscoder.py
buffer.py
upload_to_hf.py
training.py - from dictionary learning

### Analysis:

eval_crosscoder.ipynb

## ORIGINAL README
WIP repository for MATS 7.0 stream with Neel Nanda.
