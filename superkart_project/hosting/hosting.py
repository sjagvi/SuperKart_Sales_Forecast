import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id="iamsubha/SuperKart-Sales-Forecast",
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Space.")
