from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers import AutoTokenizer
import glob
import json
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import lightning as L
import torch
import math
import time
import sys
import os

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.utils import (
    get_default_supported_precision,
    chunked_cross_entropy,
    num_parameters,
)

fsdp = False

# Hyperparameters
learning_rate = 1e-3
batch_size = 16
micro_batch_size = 16
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
stable_iters = 400000
lr_decay_iters = 400000
warmup_iters = lr_decay_iters * 0.04
min_lr = 1e-4

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}
logger = None


def setup(
    devices: int = 1,
    model_name: str = "pythia-410m",
    rank: int = 0,
    precision: Optional[str] = None,
    data_dir: str = "data",
    out_dir: str = "result",
) -> None:
    precision = precision or get_default_supported_precision(training=True)
    if fsdp:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )
    fabric.print(hparams)
    fabric.launch(
        main,
        rank=rank,
        model_name=model_name,
        out_dir=Path(out_dir),
        data_dir=data_dir,
    )


def main(
    fabric: L.Fabric,
    rank: int,
    model_name: str,
    out_dir: Path,
    data_dir: str,
) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if fsdp:
        fabric.seed_everything(
            1337, workers=True
        )  # same seed for every process to init model (FSDP)
    else:
        fabric.seed_everything(workers=True)  # each process gets a different seed (DDP)


    config = Config.from_name(f"{model_name}-1024")
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    if fsdp:
        with fabric.init_module(empty_init=True):
            model = GPT(config)
    else:
        with fabric.init_module(empty_init=False):
            model = GPT(config)
    model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")


    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = load_datasets(rank=rank, data_dir=data_dir)
    train_data = IterableDatasetShard(
        train_data,
        batch_size=micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    def train_collate_fn(batch):
        return torch.tensor([sample["input_ids"] for sample in batch], device="cuda")

    def val_collate_fn(batch):
        input_ids = [
            torch.tensor(sample["input_ids"], device="cuda") for sample in batch
        ]
        labels = [torch.tensor(sample["labels"], device="cuda") for sample in batch]

        x = pad_sequence(input_ids, batch_first=True, padding_value=0)
        y = pad_sequence(labels, batch_first=True, padding_value=-1)

        max_seq_length = 1024
        if max_seq_length:
            x = x[:, :max_seq_length]
            y = y[:, :max_seq_length]

        return x, y

    train_dataloader = DataLoader(train_data, batch_size=1, collate_fn=train_collate_fn)
    val_dataloader = DataLoader(
        torch.load("lambada_openai/train-1024.pt"),
        batch_size=32,
        collate_fn=val_collate_fn,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader,
        val_dataloader,
    )
    val_dataloaders = [val_dataloader]

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    train_iter = iter(train_dataloader)
    data = []
    for step in tqdm(range(len(train_dataloader))):
        # fabric.load(resume, state)
        input_ids = next(train_iter)
        scores,loss = train(fabric, state, input_ids, val_dataloaders)
        fabric.print(f"step={step+1} | train_loss={loss:.3f}")
        data.append(
            {
                "input_ids": input_ids[0].cpu().numpy().tolist(),
                "scores": scores,
            }
        )
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "scores": Sequence(Value("float32")),
        }
    )
    processed_ds = Dataset.from_list(data, features=features)
    processed_ds.save_to_disk(out_dir / str(rank), max_shard_size="1GB", num_proc=1)


def train(fabric, state, input_ids, val_dataloaders):
    model = state["model"]
    optimizer = state["optimizer"]

    lr = get_wsd_lr(state["iter_num"]) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logits = model(input_ids)
    loss = chunked_cross_entropy(
        logits[:, :-1, :].contiguous(),
        input_ids[:, 1:].contiguous(),
        chunk_size=0,
    )
    fabric.backward(loss)
    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    return evaluate(fabric, model, val_dataloaders),loss.item()


@torch.no_grad()
def evaluate(fabric, model, val_dataloaders):
    model.eval()
    losses = []
    delta_losses=[]
    for val_dataloader in val_dataloaders:
        loss = torch.tensor(0.0, device=fabric.device)
        cnt = 0
        for input_ids, labels in val_dataloader:
            logits = model(input_ids)
            loss += chunked_cross_entropy(
                logits[:, :-1, :],
                labels[:, 1:],
                chunk_size=0,
            )
            cnt += 1
        loss = loss / cnt
        losses.append(loss.item())
    if not hasattr(evaluate,"loss_before"):
        delta_losses=[]
        evaluate.loss_before=losses
    else:
        delta_losses=[losses[i]-evaluate.loss_before[i] for i in range(len(evaluate.loss_before))]
        evaluate.loss_before=losses
    model.train()
    return delta_losses


def load_datasets(
    rank: int,
    tokenizer_name: str = "EleutherAI/pythia-410m-deduped",
    data_dir: str = "data",
    max_length: int = 1024
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    all_texts = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            json_list = json.load(f)
            for item in json_list:
                if "text" in item:
                    all_texts.append(item["text"])
    print("文本的总数是：", len(all_texts))
    chunks=[]
    for text in all_texts:
        tokenized = tokenizer(text, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        chunks.append(tokenized) 
    dataset = Dataset.from_dict({"input_ids": chunks})
    print("数据样本的总数是：", len(dataset))

    dataset = dataset.shuffle(seed=rank * 1337)

    return dataset

# learning rate decay scheduler (wsd with warmup)
def get_wsd_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it < stable_iters:
        return learning_rate
    return learning_rate * math.pow(0.5, (it - stable_iters) / 400)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
