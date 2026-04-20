# Registers the raw SuperKart dataset on the Hugging Face dataset hub.

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "iamsubha/superkart"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# create the dataset repo if it doesn't exist yet
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Created.")

# push the whole data folder
api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded.")
