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
upload_to_hf.py - Uploads model to hf, this is redundant as eval_crosscoder does this as well
training.py - from dictionary learning

compute-max-acts.sh - for validating crosscoder with acts

### Analysis:

eval_crosscoder.ipynb

### Guide to using repo:

To compute max activations 
compute_latent_activations.py --dictionary-model {CrosscoderModel}

## ORIGINAL README
WIP repository for MATS 7.0 stream with Neel Nanda.
