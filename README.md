### Create venv

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

### login to hugging face
huggingface-cli login

### login to wanb
wandb login

### Affected scripts

train-crosscoder.sh  - Some cli params are pass here   
train_crosscoder.py  - main code for training   
buffer.py  
upload_to_hf.py - Uploads model to hf, this is redundant as eval_crosscoder does this as well  
training.py - from dictionary learning, modified to take acts from buffer  

compute-max-acts.sh - for validating crosscoder with acts, not quite finished

### Analysis:

eval_crosscoder.ipynb

## ORIGINAL README
WIP repository for MATS 7.0 stream with Neel Nanda.
