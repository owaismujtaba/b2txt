'''from omegaconf import OmegaConf'''
'''
from rnn_trainer import BrainToTextDecoder_Trainer
import yaml

with open('/workspace/work/b2txt/model_training/rnn_args.yaml', 'r') as file:
    args = yaml.safe_load(file)


trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
'''


import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import yaml
from rnn_trainer import BrainToTextDecoder_Trainer


def train_worker(rank, world_size, args):
    dist.init_process_group(
        backend='nccl',                  # use 'nccl' for GPU
        init_method='tcp://127.0.0.1:29500', 
        world_size=world_size, 
        rank=rank
    )
    
    trainer = BrainToTextDecoder_Trainer(args, rank=rank, world_size=world_size)
    trainer.train()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    with open('/workspace/work/b2txt/model_training/rnn_args.yaml', 'r') as file:
        args = yaml.safe_load(file)
    
    world_size = torch.cuda.device_count()
    print(f"Launching training on {world_size} GPUs")

    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)