import hydra
import torch
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    if "repo" not in config or "ckpt" not in config:
        print(
            "Usage: upload.py +repo=<huggingface repo name> +ckpt=<rel checkpoint path>"
        )
        exit(1)

    resume_path = str(config["ckpt"])
    print(f"Loading checkpoint: {resume_path} ...")
    checkpoint = torch.load(resume_path, "cpu", weights_only=False)

    if checkpoint["config"]["model"] != config["model"]:
        print(
            "Warning: Architecture configuration given in the config file is different from that "
            "of the checkpoint."
        )

    model = instantiate(checkpoint["config"]["model"])

    model.load_state_dict(checkpoint["state_dict"])

    print("Uploading the model...")

    model.push_to_hub(config["repo"])


if __name__ == "__main__":
    main()
