import os
import sys
import time
import wandb
from pathlib import Path
import shutil
import argparse

import lightning as L
import numpy as np
import torch

wd = Path(__file__).parent.parent.resolve() 
sys.path.append(str(wd))

import whisper_openAI.whisper as whisper
from lit_llama.WL_S import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
from lightning.fabric.strategies import DeepSpeedStrategy
from generate.generate_for_WL import generate
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3,help='learning rate for the model (default: 1e-3)')
parser.add_argument('--d', type=int, default=1,help='No of GPUs (default: 1)')
parser.add_argument('--pretrained_path', type=str,default= 'model/Alpaca_PY/lit-llama.pth',help='Path to Alpaca checkpoint') 
parser.add_argument('--tokenizer_path', type=str,help='Path to LLaMA tokenizer') 
parser.add_argument('--data_path', type=str,help='Path to data') 
parser.add_argument('--num_sentences', type=int, help='Path to data') 

args = parser.parse_args()
learning_rate = args.lr
pretrained_path = args.pretrained_path
tokenizer_path = args.tokenizer_path
data_path = args.data_path
num_sentences = args.num_sentences

num_epochs = 25
weight_decay = 0.02

devices = args.d
batch_size = 32 / devices 
micro_batch_size = 4 
gradient_accumulation_steps = batch_size // micro_batch_size

train_path = f'./split_data/train_{num_sentences}.pt'
val_path = f'./split_data/test_{num_sentences}.pt'

train_data = torch.load(train_path,map_location=torch.device('cpu'))
val_data   = torch.load(val_path,map_location=torch.device('cpu'))

train_data_len = len(train_data)
val_data_len = len(val_data)

print('loaded test data')

epoch_size = train_data_len // micro_batch_size // devices
max_iters = num_epochs * epoch_size 
eval_iters = val_data_len // micro_batch_size  // devices 
warmup_steps = epoch_size * 0 // devices 


max_seq_length = 2048
max_input_length = 1000


save_interval = epoch_size 
log_interval = 100
run_name = f'WL_S_{learning_rate}_{num_sentences}'
out_dir: str = 'runs/'+run_name


wandb.login()
wandb.init(
    project="cs6212",
    name=run_name,
    group=run_name,
    config={
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "weight_decay": weight_decay,
    "batch_size": (batch_size*devices),
    "micro_batch_size":micro_batch_size,
    "dataset":'gigaspeech',
    'devices':devices,
    'max_input_length':max_input_length,
    }
)


ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}

def main():
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy= "ddp"  if devices > 1 else "auto" , 
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    config = LLaMAConfig(block_size=max_seq_length)
    

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path)
    print('loaded LLaMA checkpoint')
    with fabric.init_module():
        model = LLaMA(config)

    (_, w_ck_pt) = whisper.load_model("large-v2",device='cpu')
    print('loaded Whisper checkpoint')
    for n, p in model.named_parameters():
        if 'whisper' in n :
            layer = n.split('.')[2]
            suffix = n.split('.')[-1]
            kv = n.split('.')[4].split('_')[-1]
            w_key = f'decoder.blocks.{layer}.cross_attn.{kv}.{suffix}'
            checkpoint[n] = w_ck_pt['model_state_dict'][w_key].cpu()

        
        
    with fabric.init_module():
        model.load_state_dict(checkpoint, strict=False)
    print('loaded LAMMA model')
    mark_only_adapter_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params/1e6}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, out_dir)
    wandb.finish()

    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-adapter-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
) -> None:
    
    step_count = 0 

    for iter_num in trange(max_iters):

        t0 = time.time()

        input_ids, targets, audio_features = get_batch(fabric, model, train_data)
        logits = model(input_ids, audio_features = audio_features)
        loss = loss_fn(logits, targets)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)): 
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0: 
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            
            lr = learning_rate - ((learning_rate - 1e-5)/max_iters)*(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            wandb.log({"lr": lr})

        dt = time.time() - t0
        
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt:.2f}s")
            wandb.log({"train_iter": iter_num, "train_Iter_loss": loss.item()})
            
        if (iter_num + 1) % epoch_size == 0:
            print(f"Saving adapter weights to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{int((iter_num+1)/epoch_size):06d}.pth"))

            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.barrier()
            wandb.log({"val_step": iter_num, "val_step_loss": val_loss})
            print('End of epoch ',(iter_num+1)/epoch_size)



def generate_response(model, instruction, input=""):
    tokenizer = Tokenizer("model/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output 


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")

    if len(val_data) == val_data_len :
        eval_iters =  val_data_len // micro_batch_size  // devices
    else :
        eval_iters =  epoch_size // devices

    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets, audio_features = get_batch(fabric, model, val_data)
        logits = model(input_ids, audio_features = audio_features)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()


    model.train()
    return val_loss.item()

def loss_fn(logits, targets):
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, model ,data: list):
    ix = torch.randint(len(data), (micro_batch_size,))
    input_ids = [data[i]["input_ids"][:max_input_length].type(torch.int64) for i in ix]
    labels = [data[i]["labels_with_masked_input"][:max_input_length].type(torch.int64) for i in ix]
    audio_features = [data[i]["audio_features"].type(model.dtype) for i in ix]


    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    af = torch.cat([x for x in audio_features], dim =0)

    x, y , af  = fabric.to_device((x.pin_memory(), y.pin_memory(), af.pin_memory()))

    return x, y , af



def save_model_checkpoint(fabric, model, file_path):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = file_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:

            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict)
            torch.save(state_dict, file_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict())
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    main()
    
