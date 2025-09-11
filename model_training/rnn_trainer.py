
import os
import sys
import json
import time
import torch
import random
import logging
import pathlib
import math
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torchaudio.functional as F  # for edit distance
import numpy as np
# ----------------------------
# Import your dataset and model
# ----------------------------
from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import gauss_smooth
from rnn_model import GRUDecoderAttention

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True
torch._dynamo.config.cache_size_limit = 64

# ----------------------------
# Trainer Class
# ----------------------------
class BrainToTextDecoder_Trainer:
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.logger = self.setup_logger()
        self.logger.info(" Initializing Trainner ")
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = world_size > 1
        self.logger.info(f"Using device {self.device}, rank {rank}/{world_size}")

        if self.args['seed'] != -1:
            self.set_seed(self.args['seed'])

        # ----------------------------
        # Initialize model
        # ----------------------------
        self.logger.info(f"Initialized AttentionGRU decoding model")
        self.model = GRUDecoderAttention(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            n_classes=self.args['dataset']['n_classes'],
            rnn_dropout=self.args['model']['rnn_dropout'],
            input_dropout=self.args['model']['input_network']['input_layer_dropout'],
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
        ).to(self.device)

        self.logger.info(self.model)

        if self.multi_gpu:
            self.model = DDP(self.model, device_ids=[rank])

        self._log_param_info()

        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf

        # ----------------------------
        # Create datasets & loaders
        # ----------------------------
        self.create_datasets_and_loaders()

        # ----------------------------
        # Optimizer, Scheduler, Loss
        # ----------------------------
        self.optimizer = self.create_optimizer()
        self.learning_rate_scheduler = self.create_lr_scheduler()
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)

        # ----------------------------
        # Load checkpoint if needed
        # ----------------------------
        if self.args.get('init_from_checkpoint', False):
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

    def _log_param_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")


    # ----------------------------
    # Logger and Seed
    # ----------------------------
    def setup_logger(self):
        logger = logging.getLogger(f"Rank{self.rank}")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ----------------------------
    # Dataset and Loader Setup
    # ----------------------------
    def create_datasets_and_loaders(self):
        feature_subset = self.args['dataset'].get('feature_subset', None)
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_train.hdf5')
                            for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_val.hdf5')
                          for s in self.args['dataset']['sessions']]

        train_trials, _ = train_test_split_indicies(train_file_paths, test_percentage=0, seed=self.args['dataset']['seed'])
        _, val_trials = train_test_split_indicies(val_file_paths, test_percentage=1, seed=self.args['dataset']['seed'])

        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank,
                                           shuffle=self.args['dataset']['loader_shuffle'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=None, sampler=train_sampler,
                                       num_workers=self.args['dataset']['num_dataloader_workers'], pin_memory=True)

        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split='test',
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=None, sampler=val_sampler, num_workers=0, pin_memory=True)

        self.logger.info("Datasets and loaders initialized")

    # ----------------------------
    # Optimizer & Scheduler
    # ----------------------------
    def create_optimizer(self):
        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        param_groups = [
            {'params': bias_params, 'weight_decay': 0},
            {'params': day_params, 'lr': self.args['lr_max_day'], 'weight_decay': self.args['weight_decay_day']},
            {'params': other_params}
        ] if day_params else [
            {'params': bias_params, 'weight_decay': 0},
            {'params': other_params}
        ]

        return torch.optim.AdamW(param_groups, lr=self.args['lr_max'], betas=(self.args['beta0'], self.args['beta1']),
                                 eps=self.args['epsilon'], weight_decay=self.args['weight_decay'], fused=True)

    def create_lr_scheduler(self):
        if self.args['lr_scheduler_type'] == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            return self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid scheduler type {self.args['lr_scheduler_type']}")

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)

    # ----------------------------
    # Checkpoints
    # ----------------------------
    def load_model_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint.get('val_PER', torch.inf)
        self.best_val_loss = checkpoint.get('val_loss', torch.inf)
        self.logger.info(f"Loaded checkpoint from {path}")

    def save_model_checkpoint(self, path, PER, loss=''):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': PER,
            'val_loss': loss
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    # ----------------------------
    # Data Transform
    # ----------------------------
    def transform_data(self, features, n_time_steps, mode='train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps


    # ----------------------------
    # Training Loop
    # ----------------------------
    

    def train(self):
        self.logger.info("Starting Training process")
        for epoch in range(self.args['dataset']['num_epochs']):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            running_loss = 0.0
            
            # Wrap train_loader with tqdm
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.args['dataset']['num_epochs']}")
            
            for i, batch in loop:
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)

                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')
                logits = self.model(features, day_indicies)
                adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)
                loss = self.ctc_loss(torch.permute(logits.log_softmax(2), [1, 0, 2]), labels, adjusted_lens, batch['phone_seq_lens'].to(self.device))
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                if self.args['grad_norm_clip_value'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['grad_norm_clip_value'])
                self.optimizer.step()
                self.learning_rate_scheduler.step()

                running_loss += loss.item()
                
                # Update tqdm postfix with current loss and average loss
                loop.set_postfix({
                    'Batch Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss / (i+1):.4f}'
                })

            # Validation after each epoch
            val_metrics = self.validation(self.val_loader)
            self.logger.info(f"Epoch {epoch+1} validation PER: {val_metrics['avg_PER']:.4f}")

            val_per = val_metrics['avg_PER']
            if val_per < self.best_val_PER:
                self.best_val_PER = val_per  # Update best PER
                self.logger.info("Checkpointing model")
                self.save_model_checkpoint(
                    path=f"{self.args['checkpoint_dir']}/best_checkpoint",
                    PER=self.best_val_PER,
                )


            
    # ----------------------------
    # Validation Loop
    # ----------------------------
    def validation(self, loader):
        self.model.eval()
        total_edit_distance, total_seq_length = 0, 0
        day_per = {d: {'total_ed': 0, 'total_len': 0} for d in range(len(self.args['dataset']['sessions']))}
        with torch.no_grad():
            for batch in loader:
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)

                features, n_time_steps = self.transform_data(features, n_time_steps, 'val')
                logits = self.model(features, day_indicies)
                preds = torch.argmax(logits, dim=-1)

                for b in range(len(preds)):
                    pred_seq = preds[b][:batch['phone_seq_lens'][b]].cpu().numpy()
                    true_seq = labels[b][:batch['phone_seq_lens'][b]].cpu().numpy()
                    ed = F.edit_distance(torch.tensor(pred_seq), torch.tensor(true_seq))
                    total_edit_distance += ed
                    total_seq_length += len(true_seq)
                    day_per[day_indicies[b].item()]['total_ed'] += ed
                    day_per[day_indicies[b].item()]['total_len'] += len(true_seq)

        avg_PER = total_edit_distance / total_seq_length
        return {'avg_PER': avg_PER, 'day_PER': {d: day_per[d]['total_ed'] / day_per[d]['total_len'] for d in day_per}}



























