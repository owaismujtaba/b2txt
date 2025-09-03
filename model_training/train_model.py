from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load('/home/owaismujtaba/work/b2txt/model_training/rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()