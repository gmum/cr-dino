# Adapted from: https://github.com/facebookresearch/dino/blob/main/main_dino.py
import os
import sys
import math
from typing import Callable

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as torchvision_models
from tqdm import tqdm

import utils.dino_utils as utils
from nn.vit_dino import DINOHead
from nn.vit_dino import vit_base
from utils.dino_utils import load_pretrained_weights
from utils.mask_generators import NoMask
from utils.torch_utils import seed_worker
from utils.train_utils import get_next_batch

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


@gin.configurable()
class DINOTrainingProcedure:
    def __init__(self,
                 model_name: str,
                 arch: str = 'vit_base',
                 patch_size: int = 16,
                 output_dim: str = 65536,
                 norm_last_layer: bool = True,
                 momentum_teacher: float = 0.996,
                 use_bn_in_head: bool = False,
                 warmup_teacher_temp: float = 0.04,
                 teacher_temp: float = 0.04,
                 warmup_teacher_temp_epochs: int = 30,
                 use_fp16: bool = True,
                 weight_decay: float = 0.04,
                 weight_decay_end: float = 0.4,
                 clip_grad: float = 3.0,
                 batch_size: int = 64,
                 epochs: int = 100,
                 epoch_len: int = 100,
                 freeze_last_layer: int = 0,
                 lr: float = 0.0005,
                 warmup_epochs: int = 0,
                 min_lr: float = 1e-6,
                 optimizer: Optimizer = None,
                 drop_path_rate: float = 0.1,
                 local_crops_number: int = 8,
                 model_saving_path: str = '',
                 tensorboard_dir: str = '',
                 num_workers: int = 8,
                 valid_length: int = 2048,
                 valid_interval: int = 1,
                 log_interval: int = 1,
                 save_interval: int = 10,
                 grad_from_block: int = 11, 
                 dataset: Dataset = None,
                 seed: int = 109,
                 apply_masking: bool = False,
                 mask_generator: Callable = None,
                 pe_function: Callable = None,

        ):
        """
        Finne-tuning procedure for DINO. Starts with DINO trained model.
            Args:
            model_name: name for model saving and logging
            arch: Name of architecture to train, options: ['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
            patch_size: size of patch to embedd, default: 16 (the smaller, the longer the training)
            output_dim: imensionality of the DINO head output,
            norn_last_layer: whether or not to weight normalize the last layer of the DINO head (best:  False with vit_small and True with vit_base),
            momentum_teacher: base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. It is recommended
            to sesettting a higher value with small batches: for example use 0.9995 with batch size of 256,
            use_bn_in_head: wWhether to use batch normalizations in projection head (Default: False)
            warmup_teacher_temp: initial value for the teacher temperature: 0.04 works well in most cases
            teacher_temp: fFinal value (after linear warmup)         of the teacher temperature. For most experiments, anything above 0.07 is unstable,
            warmup_teacher_temp_epochs: number of warmup epochs for the teacher temperature (Default: 30)
            use_fp16: whether or not to use half precision for training. Improves training time and memory requirements,
                but can provoke instability and slight decay of performance. We recommend disabling
                mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.
            weight_decay: initial value of the weight decay. With ViT, a smaller value at the beginning of training works well
            weight_decay_end: final value of the weight decay. Fb uses a cosine schedule for WD and using a larger decay by
                the end of training improves performance for ViTs.
            clip_grad: maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
                help optimization for larger ViT architectures. 0 for disabling,
            batch_size: size of a batch,
            epochs: number of epochs of training,
            epoch_len: how many batches go through one epoch,
            freeze_last_layer: number of epochs during which we keep the output layer fixed. Set to 0 because we are using model for fine-tuning.
            lr: learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled
                with the batch size, and specified here for a reference batch size of 256,
            warmup_epochs: number of epochs for the linear learning-rate warm up, set to 0 for fine-tuning,
            min_lr: target LR at the end of optimization. FB uses a cosine LR schedule with linear warmup,
            optimizer: optimizer.choices=['adamw', 'sgd', 'lars'],
            drop_path_rate: stochastic depth rate
            global_crops_scale: dcale range of the cropped image before resizing, relatively to the origin image.
                Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example),
            local_crops_number: number of small local views to generate. Set this parameter to 0 to disable multi-crop training.
                When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1."
            local_crops_scale: dcale range of the cropped image before resizing, relatively to the origin image.
                Used for small local view cropping of multi-crop.
            model_saving_path: Path where to save the model
            tensorboard_dir: Path for tensorboard logging
            num_workers: number of workers per GPU
            valid_length: length of validation dataset
            valid_interval: how often to run validation
            log_interval: how often to log results
            save_interval: how often to save model
            grad_from_block: freeze student model until that block
            dataset: DataSet for training, will be split into train and valid sets
            collate_fn: collate function for dataset
            seed: value to fix random seeds
        """
        utils.fix_random_seeds(seed)
        cudnn.benchmark = True

        self.epochs = epochs
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        self.clip_grad = clip_grad
        self.epoch_len = epoch_len
        self.freeze_last_layer = freeze_last_layer
        self.drop_path_rate = drop_path_rate
        self.model_saving_path = f'{model_saving_path}/{model_name}/'
        if not os.path.exists(self.model_saving_path):
            os.makedirs(self.model_saving_path)
        self.valid_interval = valid_interval
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.student = vit_base(apply_masking=apply_masking, pe_function=pe_function)
        load_pretrained_weights(self.student, '', 'student', arch, patch_size)
        self.teacher = vit_base(pe_function=pe_function)
        load_pretrained_weights(self.teacher, '', 'teacher', arch, patch_size)

        embed_dim = self.student.embed_dim


        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = utils.MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            output_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        ))
        self.teacher = utils.MultiCropWrapper(
            self.teacher,
            DINOHead(embed_dim, output_dim, use_bn_in_head),
        )
        # move networks to gpu
        self.student, self.teacher = self.student.cuda(), self.teacher.cuda()
        # synchronize batch norms (if any)
        if utils.has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

            self.teacher_without_ddp = self.teacher
        else:
            self.teacher_without_ddp = self.teacher
        # teacher and student start with the same weights
        self.teacher_without_ddp.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Freeze student's layers up to selected block
        for m in self.student.parameters():
            self.student.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in self.student.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[2])
                if block_num >= grad_from_block:
                    m.requires_grad = True
        print(f"Student and Teacher are built: they are both {arch} network.")

        train_set, valid_set = torch.utils.data.random_split(dataset, [len(dataset) - valid_length, valid_length])
        print(f'Build train and valid dataset. Train: {len(train_set)}, valid: {len(valid_set)}')
        g = torch.Generator()
        g.manual_seed(0)
        self.data_loaders = {
            'train': DataLoader(
                dataset=train_set,  
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers),
            'valid': DataLoader(
                dataset=valid_set, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=g)
        }
        self.valid_steps = valid_length//batch_size

        # ============ preparing loss ... ============
        self.dino_loss = DINOLoss(
            output_dim,
            local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            epochs,
        ).cuda()

        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(self.student)
        self.optimizer = optimizer(params_groups)
        self.fp16_scaler = None
        if use_fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()

        # ============ init schedulers ... ============
        self.lr_schedule = utils.cosine_scheduler(
            lr * (batch_size / 256.),  # linear scaling rule
            min_lr,
            epochs, len(self.data_loaders['train']),
            warmup_epochs=warmup_epochs,
        )
        self.wd_schedule = utils.cosine_scheduler(
            self.weight_decay,
            self.weight_decay_end,
            self.epochs, len(self.data_loaders['train']),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1, epochs, len(self.data_loaders['train']))
        print(f"Loss, optimizer and schedulers ready.")
        
        timestamp = str(datetime.now()).replace(' ', '_')
        self.model_name = f'{timestamp}_{model_name}'
        self.logger = SummaryWriter(f'{tensorboard_dir}/{self.model_name}/')

        self.generate_mask = mask_generator
        if not apply_masking:
            self.generate_mask = NoMask()

    def run(self):

        # ============ optionally resume training ... ============
        self.start_iteration = 0
        print("Starting DINO training !")
        self.generators = {
            'train': iter(self.data_loaders['train']),
            'valid': iter(self.data_loaders['valid'])
            }
        bar_tqdm = tqdm(
            range(self.start_iteration, self.start_iteration + self.epochs),
            desc=f'Train loop: inf',
            initial=self.start_iteration,
            total=self.start_iteration + self.epochs,
            position=0,
            leave=True,
        )
        for epoch in bar_tqdm:
            # ============ training one epoch of DINO ... ============
            train_stats = self.step('train', epoch)

            # ============ Log and save for training ============
            if epoch % self.save_interval == 0:
                save_dict = {
                    'student': self.student.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'dino_loss': self.dino_loss.state_dict(),
                }
                if self.fp16_scaler is not None:
                    save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()
                utils.save_on_master(save_dict, os.path.join(self.model_saving_path, f'checkpoint_{epoch}.pth'))
            if epoch % self.log_interval == 0:
                self.logger.add_scalar(f'loss_train', train_stats['loss'], epoch)
            bar_tqdm.set_description(f"Train loss: {train_stats['loss']}")

            # valid pass
            if epoch % self.valid_interval == 0:
                valid_stats = self.step('valid', epoch)
                self.logger.add_scalar(f'loss_valid', valid_stats['loss'], epoch)


    def step(self, mode: str, iteration: int):

        for it in range(self.epoch_len):
            # update weight decay and learning rate according to their schedule
            it = len(self.data_loaders['train']) * iteration + it  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[it]

            # get images and move images to gpu
            next_batch, self.generators[mode] = get_next_batch(self.generators[mode], self.data_loaders[mode])
            # sample mask]
            mask = [self.generate_mask(x=iteration).cuda(non_blocking=True) for _ in next_batch]
            images = [im.cuda(non_blocking=True) for im in next_batch]
            images_teacher, images_student = images[:2], images
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                teacher_output = self.teacher(images_teacher)  # only the 2 global views pass through the teacher
                student_output = self.student(images_student, mask=mask) # masking
                loss = self.dino_loss(student_output, teacher_output, iteration)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # student update
            self.optimizer.zero_grad()
            param_norms = None
            if self.fp16_scaler is None:
                loss.backward()
                if self.clip_grad:
                    param_norms = utils.clip_gradients(self.student, self.clip_grad)
                utils.cancel_gradients_last_layer(iteration, self.student,
                                                self.freeze_last_layer)
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                if self.clip_grad:
                    self.fp16_scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(self.student, self.clip_grad)
                utils.cancel_gradients_last_layer(iteration, self.student,
                                                self.freeze_last_layer)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(self.student.parameters(), self.teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        return {'loss': loss.item()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

