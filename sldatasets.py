import argparse
import glob
import pandas as pd

# global definition
from definition import *

import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import math
from PIL import Image
import numpy as np
import decord
from transformers import VideoMAEImageProcessor, AutoTokenizer
from fairseq import checkpoint_utils, options, tasks

from vidaug import augmentors as va
from augmentation import *
import utils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from av_hubert.avhubert import hubert_pretraining, hubert #hubert启动

decord.bridge.set_bridge('torch')  # 让 decord 返回 torch Tensor


class Sign2TextDataset(Dataset):
    def __init__(self, root_dir, sign_dir, tokenizer, max_txt=128, phase='train', use_context=False, use_demo=False):
        if 'Phonexi-2014T' in sign_dir:
            self.data_name = 'Phonexi-2014T'
        else:
            self.data_name = 'How2Sign'
        # Load annotations
        #anno_path = os.path.join(root_dir, f'{phase}_info_ml.npy')
        anno_path = os.path.join(sign_dir, phase, f"{phase}.csv")
        self.spatial_dir = os.path.join(sign_dir, phase, 'spatial')
        self.lip_dir = os.path.join(sign_dir, phase, 'lip')
        if self.data_name == 'How2Sign':
            self.data = pd.read_csv(anno_path, sep='\t')
        else:
            self.data = pd.read_csv(anno_path)
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        #self.data = np.load(anno_path, allow_pickle=True).item()

        self.tokenizer = tokenizer
        self.phase = phase
        self.max_txt = max_txt
        self.use_demo = use_demo
        self.use_context = use_context
        self.frame_sample_rate = 1
        self.max_frame_len = 512
        _, _, task = checkpoint_utils.load_model_ensemble_and_task(["pretrain_models/base_lrs3_iter5.pt"])
        self.lip_transform = utils.Compose([
            utils.Normalize(0.0, 255.0),
            utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])

        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        sometimes = lambda aug: va.Sometimes(0.5, aug)
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10)),
        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
        ])

    def __len__(self):
        #return len(self.data)-1
        return len(self.data)

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by ensuring it ends with a period.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text.lower()

    def pad_video(self, video, target_len):
        F, H, W = video.shape
        result = torch.zeros((target_len, H, W), dtype=torch.float32)
        result[:F] = torch.from_numpy(video)
        return result

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        text = self._normalize_text(sample['text'])
        name = sample['filename']

        spatial_path = os.path.join(self.spatial_dir, name + '_s2wrapping.npy')
        if not os.path.exists(spatial_path):
            raise FileNotFoundError(f"Visual file not found: {spatial_path}")
        spatial_feat = torch.tensor(np.load(spatial_path))  # [T, C, H, W] or [T, D]

        # 设置帧率
        #fps = 25

        # 提前过滤空特征
        if spatial_feat.shape[0] == 0:
            print(f"[Warning] Empty spatial feature at idx {idx}, file: {spatial_path}")
            return None

        # 获取帧数
        num_frames = math.ceil(len(spatial_feat) / self.frame_sample_rate)
        pval = spatial_feat[::self.frame_sample_rate]

        if num_frames > self.max_frame_len:
            num_frames = self.max_frame_len
            if pval.size(0) > self.max_frame_len:
                start_index = random.randint(0, pval.size(0) - self.max_frame_len)
            else:
                start_index = 0
            spatial_feat = pval[start_index:start_index + self.max_frame_len]
        else:
            spatial_feat = pval

        # 加载嘴型视频
        if self.data_name == 'Phonexi-2014T':
            lip_name = 'mouth_' + name + '_output.mp4'
        else:
            lip_name = 'mouth_' + name + '.mp4'
        lip_path = os.path.join(self.lip_dir, lip_name)

        try:
            lip_frames = utils.load_lip(lip_path, num_frames)
            lip_tensor = self.lip_transform(lip_frames)
        except Exception as e:
            print(f"[Warning] Failed to load lip video: {lip_path}, error: {e}")
            return None

        lip_length = lip_tensor.shape[0]

        '''# ✅ 过滤帧数不足 1 秒的视频
        if spatial_feat.shape[0] < fps or spatial_feat.shape[0]>256:
            print(f"[Info] Skipping short sample idx {idx} ({spatial_feat.shape[0]} frames, {lip_length} lip frames)")
            return None'''

        # Tokenize text
        target = self.tokenizer(
            text_target=text,
            padding="max_length",
            max_length=self.max_txt,
            truncation=True,
            return_tensors="pt"
        )

        outputs = {
            'spatial_feat': spatial_feat,  # [T, C, H, W] or [T, D]
            'lip_video': lip_tensor,
            'num_frames': spatial_feat.shape[0],
            'lip_length': lip_length,
            'labels': target.input_ids.squeeze(0),
            'attention_mask': target.attention_mask.squeeze(0),
        }

        # 处理 context（可选）
        context = None
        if self.use_context:
            en_text = "on sunday in the northwest, a mixture of sun and clouds with some of the partly thundering showers."
            fr_text = "dimanche dans le nord-ouest, un mélange de soleil et de nuages avec certaines des averses partiellement tonitruantes."
            texts = "am sonntag im nordwesten eine mischung aus sonne und wolken mit einigen zum teil gewittrigen schauern."
            _ex_lang_trans = [f"{en_text}={texts}", f"{fr_text}={texts}"]
            context = ' '.join(_ex_lang_trans)

        outputs['context'] = context

        return outputs

    def __str__(self):
        return f'#total {self.phase} set: {len(self.data)}.'

    def collate_fn(self, batch):
        # 过滤掉 num_frames 或 lip_length 太短的样本
        filtered_batch = [b for b in batch if b is not None]
        if len(filtered_batch) == 0:
            return None

        spatial_feat = [item['spatial_feat'] for item in filtered_batch]
        num_frames = [item['num_frames'] for item in filtered_batch]
        lip_length = [item['lip_length'] for item in filtered_batch]
        context = [item['context'] for item in filtered_batch]

        padded_spatial = pad_sequence(spatial_feat, batch_first=True)

        max_len = max(6, max(item['lip_video'].shape[0] for item in filtered_batch))
        lip_video = torch.stack([
            self.pad_video(item['lip_video'], max_len) for item in filtered_batch
        ])

        collated_batch = {
            'spatial_feat': padded_spatial,  # [B, T, C, H, W]
            'lip_video': lip_video,
            'context': context,
            'num_frames': torch.tensor(num_frames),
            'lip_length': torch.tensor(lip_length),
            'labels': torch.stack([item['labels'] for item in filtered_batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in filtered_batch]),
        }
        return collated_batch


def get_args_parser():
    parser = argparse.ArgumentParser('Sign language LLM', add_help=False)

    parser.add_argument('--root_dir', default='data/Phonexi-2014T',
                        help='root path')
    parser.add_argument('--sign_dir',
                        default='/mnt/ceph-hdd/cold/nim00016/data/How2sign',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--phase', default='test',
                        help='[train, dev, test]')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--max_frames', default=32, type=int)

    # model params
    parser.add_argument('--t5_model', default='/mnt/ceph-hdd/cold/nim00016/huggingface/flan-t5-xl',
                        help='t5 path')
    parser.add_argument('--max_txt', default=128, type=int)


    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sign language dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    print("Arguments parsed successfully")  # 调试输出

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.t5_model,
        max_length=args.max_txt,
    )
    print("Tokenizer loaded successfully")  # 调试输出

    print(f"Creating dataset from {args.phase}")  # 调试输出
    dev_data = Sign2TextDataset(args.root_dir, args.sign_dir, tokenizer, args.max_txt, args.phase, use_context=True)
    print("Dataset created successfully")  # 调试输出
    print(dev_data)
    sample = dev_data[5]
    samples = dev_data[0]
    print(f"lip_video1 {samples['lip_video'].shape}")
    print(f"spatial_feat {sample['spatial_feat'].shape}")
    print(f"lip_video {sample['lip_video'].shape}")

    print(f"num_frames {sample['num_frames']}")
    print(f"lip_length {sample['lip_length']}")
    print(f"labels: {sample['labels'].shape}")
    out_text = tokenizer.batch_decode(sample['labels'].unsqueeze(0), skip_special_tokens=True)
    print(f"out_text {out_text}")

    print(f"context: {sample['context']}")
    print(f"attention_mask: {sample['attention_mask'].shape}")
    empty_samples = []
    for i in range(len(dev_data)):
        try:
            sample = dev_data[i]
            if sample['spatial_feat'].shape[0] == 0:
                print(f"[Warning] Empty spatial feature at index {i}, file: {dev_data.data.iloc[i]['filename']}")
                empty_samples.append(i)
        except Exception as e:
            print(f"[Error] index {i} failed with error: {e}")
            empty_samples.append(i)

    print(f"Total empty samples: {len(empty_samples)}")
    if empty_samples:
        print("Problematic indices:", empty_samples[:20], "..." if len(empty_samples) > 20 else "")

    data_loader = DataLoader(
        dev_data,
        batch_size=16,
        shuffle=(args.phase == "dev"),
        num_workers=4,
        collate_fn=dev_data.collate_fn  # 这里很关键
    )
    batch = next(iter(data_loader))
    print("Batch loaded successfully!")
    print(batch['spatial_feat'].shape)






