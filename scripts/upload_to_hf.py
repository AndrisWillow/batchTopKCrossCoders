from huggingface_hub import HfApi, Repository
from pathlib import Path
import shutil

main_dir = Path(__file__).parent.parent.resolve()
SAVE_DIR = main_dir / "checkpoints"

model_name = "Qwen2.5-0.5B-L13-k50-lr5e-05-CCLoss"
full_model_name = f"AndrisWillow/{model_name}" # to upload
local_model_dir = Path(f"{SAVE_DIR}/{model_name}")
files_to_upload = ["model_final.pt", "last_eval_logs.pt", "config.json"]

api = HfApi()
api.create_repo(repo_id=model_name, exist_ok=True)

hf_dir = Path("/tmp/hf_upload")
if hf_dir.exists():
    shutil.rmtree(hf_dir)
repo = Repository(local_dir=hf_dir, clone_from=full_model_name)

for fname in files_to_upload:
    src = local_model_dir / fname
    dst = hf_dir / fname
    shutil.copy(src, dst)

repo.push_to_hub(commit_message="Added BatchTopK Crosscoder")
