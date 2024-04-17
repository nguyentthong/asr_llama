
import json
import os
import sys
import time
import wandb
from pathlib import Path
import shutil
import argparse
import tqdm
from evaluate import load

wd = Path(__file__).parent.parent.resolve() 
sys.path.append(str(wd))

import lightning as L
import numpy as np
import torch
import whisper_openAI.whisper as whisper

from lit_llama.WL_M import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_from_state_dict
from lit_llama.tokenizer import Tokenizer
from lightning.fabric.strategies import DeepSpeedStrategy
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from generate.generate_for_WL import generate 
wer = load("wer")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='Path to the directory containing the Adapter checkpoints')
parser.add_argument('--save_dir', default= './wers',type=str, help='directory to save predictions as JSON files')
parser.add_argument('--data_path',type=str, help='Path to dataset to run inference on')
parser.add_argument('--alpaca_path',type=str, help='Path to the Alpaca checkpoints')
parser.add_argument('--tokenizer_path',type=str, help='Path to the tokenizer')


args = parser.parse_args()

save_dir = args.save_dir
root_path = args.root
data_path = args.data_path
pretrained_path = args.pretrained_path
tokenizer_path = args.tokenizer_path

files = os.listdir(root_path)
files.sort() 


torch.set_float32_matmul_precision("high")
quantize  = None
pretrained_path: Path = Path(pretrained_path)
tokenizer_path: Path = Path(tokenizer_path)
tokenizer = Tokenizer(tokenizer_path)
data   = torch.load(data_path,map_location=torch.device('cpu'))

assert pretrained_path.is_file()
assert tokenizer_path.is_file() 


fabric = L.Fabric(devices=1)
dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


print("Loading model ...", file=sys.stderr)

checkpoint = torch.load(pretrained_path) 
print('loaded LLaMA checkpoint')

config = LLaMAConfig(block_size=2048)
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

print('eveything except llama model loaded')


def result(adapter_path,model):

    t0 = time.time()
    adapter_checkpoint = torch.load(adapter_path) 
    print('loaded Adapter checkpoint')
    with fabric.init_module():
        model.load_state_dict(adapter_checkpoint, strict=False)
    model.to(dtype)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)


    c = 0
    return_dict = {}
    pr = []
    gt = []
    to_json = []
    
    for datapoint in tqdm.tqdm(data):
        encoded = datapoint['input_ids_no_response'].to(model.device)
        audio_features = datapoint['audio_features'].to(model.device).to(dtype)
        ground_truth =  datapoint['ground_truth']

        y = generate(model =model, idx = encoded, max_new_tokens =150,max_seq_length=2048,temperature=0.2, top_k=1, eos_id=tokenizer.eos_id, audio_features= audio_features)

        model.reset_cache()
        output = tokenizer.decode(y)
        inf = output[len(tokenizer.decode(encoded)):].split('\n')[0].strip()
        ref = ground_truth.strip()
        
        if inf == ref: 
            c = c + 1
        pr.append(inf)
        gt.append(ref)
        to_json.append({'inference':inf, 'ground_truth':ref})

    print(f'For {adapter_path}')
    return_dict['adapter_path']=adapter_path
    wer_ = wer.compute(predictions=pr, references=gt)
    print(f'WER is {wer_}')
    return_dict['WER']=wer_
    print(f'Ground truth matches is {c}/{len(data)}')
    to_json.append({'wer':wer_, 'gtms':f'{c}/{len(data)}'})
    return_dict['gtms']=c/len(data)
    

    with open(os.path.join(save_dir,adapter_path.split('/')[-2]+adapter_path.split('/')[-1]+'.json'),'w') as f:
        f.write(json.dumps(to_json , indent = 4,ensure_ascii=False))
    print(os.path.join(save_dir,adapter_path.split('/')[-2]+'.json'))
    
    print('the post string normalization wer is')
    x = 0
    for i in range(len(pr)):
        pr[i] = pr[i].lower().replace('.','').replace(',','').replace('-','').replace('?','').replace("'",'')
        gt[i] = gt[i].lower().replace('.','').replace(',','').replace('-','').replace('?','').replace("'",'')
        if pr[i] == gt[i]:
            x = x+1
    post_wer =wer.compute(predictions=pr, references=gt)
    print('WER',post_wer)
    return_dict['post_ST_wer']=post_wer
    print(x,'/',len(pr))
    return_dict['post_gtms']=x/len(pr)
    print('*********************')
    return return_dict


wandb.login()
wandb.init(
    project="WHISPERing LLaMA", 
    name=root_path,
)


for i in files:
    import pdb
    pdb.set_trace()
    adapter_path = os.path.join(root_path,i)
    try:
        result_dict = result(adapter_path,model)
        wer_percent = result_dict['WER']*100
        wer_percent_post = result_dict['post_ST_wer']*100
        
        gt_percent = result_dict['gtms']*100
        gt_percent_post = result_dict['post_gtms']*100
        wandb.log({'epoch':i,
                'WER':wer_percent,
                "WER_post":wer_percent_post,
                "GTM":gt_percent,
                "GTM_post":gt_percent_post})
    except:
        print('skippin',adapter_path)
