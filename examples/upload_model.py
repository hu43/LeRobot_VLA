from huggingface_hub import HfApi

api = HfApi()

repo_id = "hu43/lerobot_zihao_model_a"

api.upload_folder(
    folder_path="~/hu/lerobot/output_lerobot_train/shake/checkpoints/last/pretrained_model",
    repo_id=repo_id,
    repo_type="model"
)

api.create_tag(repo_id, tag="v0.1.0", repo_type="model")