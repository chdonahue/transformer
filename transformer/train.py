import torch
import os
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformer.dataset import BilingualDataset, causal_mask
from transformer.models.model import build_transformer
from dotenv import load_dotenv
from omegaconf import OmegaConf


from transformer.config import get_weights_file_path, get_config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import warnings


def setup_wandb(config: dict):
    """Initialize WandB with configuration."""
    # Load environment variables from .env file
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # Start a new wandb run:
    return wandb.init(
        project=config.wandb.project_name,
        name=config.wandb.run_name,
        config=OmegaConf.to_container(config.model),
        tags=config.wandb.tags,
    )


def beam_search_decode():
    """
    Not yet implemented
    """
    raise ValueError("Not Yet Implemented")


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    # Precompute encoder output. Reuse for all tokens from decoder:
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder SOS:
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target:
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate output of decoder:
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token:
        prob = model.project(out[:, -1])
        # select token with max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    loss_fn,
    num_examples=2,
):
    model.eval()  # model in evaluation mode
    total_val_loss = 0.0
    val_batches = 0
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print to console
            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break

            # Calculate validation loss
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            val_loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            total_val_loss += val_loss.item()
            val_batches += 1

    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
    return avg_val_loss


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
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


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )
    # Build tokenizers:
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 90/10 split:
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
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    # Define device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} Parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # training loop:
    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)  # (Batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)  # (Batch, 1, 1 seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch, 1, seq_len, seq_len)

            # Run tensors through transformer:
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (Batch, seq_len, d_model)
            proj_output = model.project(
                decoder_output
            )  # (Batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (Batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) -> (Batch *seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            # Log training loss
            wandb.log(
                {"train_loss": loss.item(), "epoch": epoch, "global_step": global_step}
            )

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Backprop:
            loss.backward()

            # Update weights:
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run every Epoch:
        val_loss = run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            loss_fn,
        )

        # Log validation loss
        wandb.log({"val_loss": val_loss, "epoch": epoch, "global_step": global_step})

        # Save model after every epoch (TODO: make this a free param)
        if epoch % config.save_every_n_epochs == 0:
            print(f"Saving snapshot epoch: {epoch}")
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

    # Load config:
    config = OmegaConf.load("test_config.yaml")
    # config = get_config() # old way

    # Initialize wandb using OmegaConf:
    run = setup_wandb(config)

    # Train model:
    train_model(config.model)
