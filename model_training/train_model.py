'''from omegaconf import OmegaConf'''
from rnn_trainer import BrainToTextDecoder_Trainer
import yaml

with open('model_training/rnn_args.yaml', 'r') as file:
    args = yaml.safe_load(file)


trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
