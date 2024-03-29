from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config: dict, epoch: str):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config: dict):
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    model_filename = f'{config["model_basename"]}*'
    weight_files = list(Path(model_folder).glob(model_filename))
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return str(weight_files[-1])
