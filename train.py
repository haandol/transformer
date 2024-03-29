import warnings
from pathlib import Path

import torch
import torchmetrics
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset
from model import build_transformer
from config import latest_weights_file_path, get_weights_file_path, get_config


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds: Dataset, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: dict, ds: Dataset, lang: str):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config: dict):
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # split into train and val
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print("Max length of source language:", max_len_src)
    print("Max length of target language:", max_len_tgt)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def train_model(config: dict):
    # define device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print(f"using device: {device}")
    device = torch.device(device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload
        else get_weights_file_path(config, config["preload"]) if preload else None
    )
    if model_filename:
        print(f"preloading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("no model preloaded")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (batch_size, 1, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch_size, seq_len, d_model)
            proj_output = model.projection(
                decoder_output
            )  # (batch_size, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (batch_size, seq_len)

            # (batch_size, seq_len, tgt_vocab_size) --> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log loss
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
