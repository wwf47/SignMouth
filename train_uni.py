# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

# *transformers
from transformers import AutoTokenizer, MBartTokenizer,MBartConfig

# *user-defined
from module.models import SignClip
from sldatasets import Sign2TextDataset
import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_score import rouge_scorer

# *timm
from timm.optim import create_optimizer
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler


# global definition
from definition import *
from evaluate import load
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warm_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--root_dir', default='data/Phonexi-2014T',
                        help='root path')
    parser.add_argument('--max_frames', default=128, type=int)
    parser.add_argument('--sign_dir',
                        default='/mnt/ceph-hdd/cold/nim00016/data/Phonexi-2014T',
                        help='path where to save, empty for no saving')
    parser.add_argument('--use_demo',
                        default=False, type=bool)
    parser.add_argument('--lang',
                        default='German',
                        help='Language')
    # model params
    parser.add_argument('--llm_path', type=str,
                        default='/mnt/ceph-hdd/cold/nim00016/huggingface/flan-t5-xl')

    parser.add_argument('--max_txt', default=128, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--use_spatial', default=False, type=bool)
    parser.add_argument('--use_align', default=False, type=bool)
    parser.add_argument('--use_vlign', default=False, type=bool)
    parser.add_argument('--use_frame', default=False, type=bool)
    parser.add_argument('--use_lip', default=False, type=bool)
    parser.add_argument('--overlap', default=8, type=int)
    parser.add_argument('--warm_epoch', default=0, type=int)

    #prompt design
    parser.add_argument('--use_context', default=False, type=bool)
    parser.add_argument('--prompt_pos', default=1, type=int)

    # * visualization
    parser.add_argument('--visualize', action='store_true')

    #fusion methods
    parser.add_argument('--fusion_mode', default='joint', type=str)


    return parser

def main(args):
    train_loss_per_epoch = []
    dev_loss_per_epoch = []
    dev_bleu_per_epoch = []
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, max_length=args.max_txt)
    '''if tokenizer.pad_token is None:
        print("Tokenizer missing pad_token, using unk_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token'''


    train_data = Sign2TextDataset(root_dir=args.root_dir, sign_dir=args.sign_dir, tokenizer = tokenizer, max_txt=args.max_txt, phase='train', use_context = args.use_context, use_demo=args.use_demo)
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    shuffle=True,
                                                                    rank=torch.distributed.get_rank(),
                                                                    drop_last=False)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                  drop_last=True)
    
    
    dev_data = Sign2TextDataset(root_dir=args.root_dir, sign_dir=args.sign_dir, tokenizer = tokenizer, max_txt=args.max_txt, phase='test', use_context = args.use_context, use_demo=args.use_demo)
    print(dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)
    
    test_data = Sign2TextDataset(root_dir=args.root_dir, sign_dir=args.sign_dir, tokenizer = tokenizer, max_txt=args.max_txt, phase='test', use_context = args.use_context, use_demo=args.use_demo)
    print(test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)
    
    print(f"Creating model:")
    model = SignClip(llm_name=args.llm_path, use_lora=True, use_align=args.use_align, alpha=args.alpha, beta=args.beta, use_spatial=args.use_spatial, use_lip=args.use_lip, use_vlign=args.use_vlign, use_frame=args.use_frame, overlap=args.overlap, prompt_pos=args.prompt_pos, fusion_mode=args.fusion_mode, lang=args.lang)
    model.to(device)
    print(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)

    lr_scheduler = scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=args.min_lr,
                T_max=args.epochs,
            )
    '''lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,  # 总的 epoch 数 (余弦周期)
        lr_min=args.min_lr,  # 最小学习率 (对应 min_lr)
        warmup_t=0,  # warmup 的 epoch 数 (比如 5 个 epoch)
        warmup_lr_init=args.warm_lr,  # warmup 起始学习率
        warmup_prefix=True,  # warmup 结束后才进入余弦衰减
        cycle_limit=1,  # 只跑一个周期（不重启）
        t_in_epochs=True  # 按 epoch 更新（如果想 step-level 更新可以设成 False）
    )'''
    loss_scaler = NativeScaler()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.2, num_classes=2454)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        print('Resuming Model Parameters... ')
        load_path = os.path.join(args.resume, 'best_checkpoint.pth')
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, args.epochs, 'dev')
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f} ")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, args.epochs, 'test')
        print(f"BLEU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu4']:.2f}")
        print(f"BLEU-3 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu3']:.2f}")
        print(f"BLEU-2 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu2']:.2f}")
        print(f"BLEU-1 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu1']:.2f}")
        print(f"RougeL of the network on the {len(test_dataloader)} test videos: {test_stats['rougeL']:.2f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    counter = 0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if isinstance(dev_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                dev_dataloader.sampler.set_epoch(epoch)
            if isinstance(test_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                test_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)
        lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            checkpoint_path = output_dir / f'checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

        '''if args.distributed:
            dist.barrier()'''
        #wwf916

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, epoch, 'dev')
        '''if args.distributed:
            dist.barrier()'''
        #wwf916
        # 记录平均loss和BLEU-4
        train_loss_per_epoch.append(float(train_stats['loss']))
        dev_bleu_per_epoch.append(float(test_stats['bleu4']))
        dev_loss_per_epoch.append(float(test_stats['loss']))

        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]
            if utils.is_main_process():
                best_ckpt_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, best_ckpt_path)
                time.sleep(1)
                counter = 0
        else:
            counter += 1
            if counter>=args.patience_epochs:
                print(f"Early stopping at epoch {epoch}")
                break
            
        print(f'Max BLEU-4: {max_accuracy:.2f}%')
        '''if utils.is_main_process():
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/Bleu_4':test_stats['bleu4'], 'dev/Best_Bleu_4': max_accuracy})'''

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        if args.distributed:
            dist.barrier()

        best_ckpt_path = Path(args.output_dir) / 'best_checkpoint.pth'

        # 只让 rank0 检查文件
        if utils.is_main_process():
            if not best_ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")
            while best_ckpt_path.stat().st_size < 1000:  # 文件太小可能还没写完
                print(f"Waiting for checkpoint to be fully written...")
                time.sleep(1)

        # 所有进程再次同步，确认 rank0 文件检查完毕
        if args.distributed:
            dist.barrier()
        checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)


        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, args.epochs, 'test')
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
        
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, args.epochs, 'test')
        print(f"BLEU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu4']:.2f}")
        print(f"BLEU-3 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu3']:.2f}")
        print(f"BLEU-2 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu2']:.2f}")
        print(f"BLEU-1 of the network on the {len(test_dataloader)} test videos: {test_stats['bleu1']:.2f}")
        print(f"RougeL of the network on the {len(test_dataloader)} test videos: {test_stats['rougeL']:.2f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 绘制Loss和BLEU曲线
    if utils.is_main_process():  # 只在主进程绘图
        epochs = list(range(1, len(train_loss_per_epoch) + 1))

        plt.figure(figsize=(12, 4))
        # Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_per_epoch, 'b-o', label='Train Loss')
        plt.plot(epochs, dev_loss_per_epoch, 'g-o', label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # BLEU 曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, dev_bleu_per_epoch, 'r-o', label='Dev BLEU-4')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-4')
        plt.title('Validation BLEU-4 Curve')
        plt.legend()

        plt.tight_layout()
        plt.savefig(str(output_dir / "loss_bleu_curve.png"))
        plt.show()


def train_one_epoch(args, model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int,
                    set_training_mode=True):
    model.train(set_training_mode)
    device = next(model.parameters()).device

    metric_logger = utils.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('lr_t5', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10

    for step, src_input in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        is_valid = torch.tensor(0 if src_input is None else 1, device=device)
        if dist.is_initialized():
            dist.all_reduce(is_valid, op=dist.ReduceOp.MIN)
        # 只要有任意一个 rank 遇到 None，大家都跳过这一步
        if is_valid.item() == 0:
            continue
            #loss = model(src_input['video'], src_input['lip_video'], src_input['labels'], src_input['attention_mask'])
        if args.warm_epoch!=0 and args.warm_epoch > epoch:

            visual_out, visual_mask, sign_out, lip_out = model.module._get_visual_outputs(src_input['spatial_feat'].cuda(), src_input['lip_video'].cuda(), src_input['num_frames'].cuda(), src_input['lip_length'])
            with torch.no_grad():
                input_embeds, attention_mask,_ = model.module._prepare_joint_inputs(visual_out, visual_mask, model.module.prompt, src_input['context'])
            con_loss = model.module.visual_textual_align(src_input['labels'].cuda(), visual_out)
            loss = con_loss
        else:
            loss = model(src_input['spatial_feat'], src_input['lip_video'].cuda(), src_input['num_frames'], src_input['lip_length'], src_input['labels'], src_input['attention_mask'], src_input['context'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        if "lr" in optimizer.param_groups[0]:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if len(optimizer.param_groups) > 1 and "lr" in optimizer.param_groups[1]:
            lr_t5_val = optimizer.param_groups[1]["lr"]
            if math.isfinite(lr_t5_val):  # 防止 nan
                metric_logger.update(lr_t5=round(float(lr_t5_val), 8))

        if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
            utils.visualization(model.module.visualize())
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def metric(predictions, references, split="train"):
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize='13a').corpus_score(predictions, [references]).score
    log_dicts["bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize='13a').corpus_score(predictions, [references]).score
            log_dicts["bleu" + str(i)] = score

        # Calculate ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]

        # Aggregate ROUGE-L scores (average precision, recall, and F1)
        avg_precision = sum(score.precision for score in rouge_scores) / len(rouge_scores)
        avg_recall = sum(score.recall for score in rouge_scores) / len(rouge_scores)
        avg_f1 = sum(score.fmeasure for score in rouge_scores) / len(rouge_scores)

        log_dicts["rougeL_precision"] = avg_precision*100
        log_dicts["rougeL_recall"] = avg_recall
        log_dicts["rougeL_f1"] = avg_f1
    return log_dicts


def evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, epoch, split):
    """
    分布式安全版评估：
    - 所有 rank 都执行 forward / generate
    - 用 all_gather 收集各 rank 的预测与参考
    - 仅 rank0 计算指标，再广播给所有 rank
    - metric_logger.synchronize_between_processes() 全员参与，避免 collective mismatch
    """
    model.eval()
    device = next(model.parameters()).device

    metric_logger = utils.MetricLogger(delimiter="  ")
    # 基础 loss
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))

    # BLEU 和 ROUGE 一般没有平滑需求，用 window_size=1 即可
    metric_logger.add_meter('bleu4', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    if split == 'test':
        metric_logger.add_meter('bleu1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('bleu2', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('bleu3', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('rougeL', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = split + ':'

    local_pres, local_refs = [], []

    with torch.no_grad():
        for step, src_input in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):

            is_valid = torch.tensor(0 if src_input is None else 1, device=device)
            if dist.is_initialized():
                dist.all_reduce(is_valid, op=dist.ReduceOp.MIN)
            if is_valid.item() == 0:
                # 保证 logger 仍然前进一步（可选）
                metric_logger.update(loss=0.0)
                continue

            # —— 将用到的张量统一搬到 device（避免不同 rank 有 .cuda() 使用差异）——
            spatial_feat  = src_input['spatial_feat'].to(device, non_blocking=True)
            lip_video     = src_input['lip_video'].to(device, non_blocking=True)
            num_frames    = src_input['num_frames'].to(device, non_blocking=True)
            lip_length    = src_input['lip_length']  # 通常是 list/int，按需处理
            labels        = src_input['labels'].to(device, non_blocking=True)
            attention     = src_input['attention_mask'].to(device, non_blocking=True)
            context       = src_input['context']     # 文本/列表，留在 CPU 即可

            # —— 计算 loss（所有 rank 都算，同步路径）——
            if args.warm_epoch != 0 and args.warm_epoch > epoch:
                # 只有 DDP 才有 .module，这里优先走 .module，单卡/非 DDP 时回退
                core = model.module if hasattr(model, "module") else model
                visual_out, visual_mask, sign_out, lip_out = core._get_visual_outputs(
                    spatial_feat, lip_video, num_frames, lip_length
                )
                input_embeds, attn_mask, _ = core._prepare_joint_inputs(
                    visual_out, visual_mask, core.prompt, context
                )
                tgt_loss = core.visual_textual_align(labels, visual_out)
            else:
                tgt_loss = model(spatial_feat, lip_video, num_frames,
                                 lip_length, labels, attention, context)

            metric_logger.update(loss=float(tgt_loss.item()))

            # —— 生成预测（所有 rank 都生成）——
            outputs = model_without_ddp.generate(
                spatial_feat, lip_video, num_frames, lip_length, context, labels
            )  # 你的 generate 已经内部在 device 上

            # 参考文本解码到小写
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
            targets = [t.lower() for t in targets]

            # 收集到本地 list，后面 all_gather
            local_pres.extend(outputs)
            local_refs.extend(targets)

    # —— 各 rank 一起 all_gather 预测与参考 —— #
    # （utils 里新增 all_gather_list / broadcast_object，见下节）
    gathered_pres = utils.all_gather_list(local_pres)
    gathered_refs = utils.all_gather_list(local_refs)

    # 只有 rank0 计算指标，然后广播给所有 rank
    if utils.is_main_process():
        dic = metric(gathered_pres, gathered_refs, split)
    else:
        dic = None

    dic = utils.broadcast_object(dic, src=0)

    # —— 用广播后的结果更新各 rank 的 logger，保持返回一致 —— #
    metric_logger.meters['bleu4'].update(dic['bleu4'])
    if split == 'test':
        metric_logger.meters['bleu1'].update(dic['bleu1'])
        metric_logger.meters['bleu2'].update(dic['bleu2'])
        metric_logger.meters['bleu3'].update(dic['bleu3'])
        metric_logger.meters['rougeL'].update(dic['rougeL_precision'])

    # 全员同步（allreduce），避免有人没调到这里
    metric_logger.synchronize_between_processes()

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        # 单卡 eval 时，原样打印
        print(dic)
        print('*' * 80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

