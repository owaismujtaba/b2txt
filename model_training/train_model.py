from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load('/home/owais/Desktop/work/nejm-brain-to-text/model_training/rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()