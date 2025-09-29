import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, MT5ForConditionalGeneration, MT5Tokenizer
from peft import LoraConfig, get_peft_model
from fairseq import checkpoint_utils, options, tasks
import utils
from module.clip_loss import clip_loss
from module.tconv import TemporalConv
from module.helpers import create_mask, derangement
import re
import numpy as np
import json
from typing import List, Union, Optional
from av_hubert.avhubert import hubert_pretraining, hubert #hubert启动

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(mm_projector_type='linear', mm_hidden_size=512, hidden_size=768, mlp_depth=1):
    if mm_projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match.group(1).isdigit() else mlp_depth
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if mm_projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {mm_projector_type}')


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_lip = nn.Linear(dim, dim)
        self.proj_sign = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, sign_feat, lip_feat):
        sign_proj = self.proj_sign(sign_feat)  # (B, T, D)
        lip_proj = self.proj_lip(lip_feat)    # (B, T, D)
        gate_input = torch.cat([sign_proj, lip_proj], dim=-1)  # (B, T, 2D)
        alpha = self.gate(gate_input)  # (B, T, 1)
        fused = alpha * lip_proj + (1 - alpha) * sign_proj
        return fused  # (B, T, D)


class LipFeatureExtractor(nn.Module):
    def __init__(self, ckpt_path='pretrain_models/base_lrs3_iter5.pt', device='cuda', feature_layer=6):
        super().__init__()
        self.device = device
        self.feature_layer = feature_layer

        # Load the model checkpoint
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

        self.model = models[0].to(device).eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.feat_dim = 768

    @torch.no_grad()
    def forward(self, mouth_rois):
        mouth_rois = mouth_rois.unsqueeze(1)
        mouth_rois = mouth_rois.to(device='cuda', dtype=torch.float32)
        assert mouth_rois.dim() == 5, "Input must be [BS, T, H, W]"
        #inputs = mouth_rois.permute(0, 2, 1, 3, 4)  # [BS, T, C, H, W]
        features, _ = self.model.extract_finetune(
            source={'video': mouth_rois, 'audio': None},
            padding_mask=None,
            output_layer=None
        )
        #print(f"feature {features.shape}")
        #pooled_features = torch.mean(features, dim=1)
        return features


class QFormer(nn.Module):
    def __init__(self, d_model=512, out_model=512, n_queries=8, n_heads=8, max_frames=512):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, out_model),
            nn.GELU(),
            nn.Linear(out_model, out_model)
        )
        # 帧级位置嵌入
        self.frame_pos_embedding = nn.Embedding(max_frames, d_model)

    def forward(self, sign_feat, lip_feat):
        # sign_feat, lip_feat: [B, T, D]
        B, T, D = sign_feat.size()

        # 添加帧级位置编码
        pos_ids = torch.arange(T, device=sign_feat.device)  # [T]
        pos_emb = self.frame_pos_embedding(pos_ids)         # [T, D]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)    # [B, T, D]

        sign_feat = sign_feat + pos_emb
        lip_feat  = lip_feat + pos_emb

        # 拼接 sign + lip
        fused_inputs = torch.cat([sign_feat, lip_feat], dim=1)  # [B, 2T, D]

        # expand queries
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Nq, D]

        # cross-attention
        fused, _ = self.cross_attn(Q, fused_inputs, fused_inputs)  # [B, Nq, D]
        fused = self.ffn(fused)
        return fused


class SignClip(nn.Module):
    def __init__(self,
                 llm_name: str = "google/flan-t5-xl",
                 fusion_mode: str = "joint",
                 input_size: int = 2048,
                 inter_hidden: int = 768,
                 use_lora: bool = True,
                 use_align: bool = True,
                 use_sign: bool = True,
                 use_spatial: bool =True,
                 use_lip: bool = True,
                 use_vlign: bool = True,
                 use_frame: bool = True,
                 prompt_pos: int = 1,
                 overlap: int = 8,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 gama: float = 0.1,
                 lang: str = "German",
                 num_queries: int = 8,
                 **kwargs):
        super().__init__()
        #self.prompt = f"Translate this weather forecast sentence into {lang}."
        #self.prompt = f"Translate the given sign language video into {lang}."
        # self.prompt = f"“Translate this sign language video into written {lang}."
        self.prompt = f"Translate the given sentence into {lang}."

        self.inter_hidden = inter_hidden
        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.input_size = input_size
        self.llm_name = llm_name
        self.use_frame = use_frame

        if 'mt5' in llm_name:
            self.llm = MT5ForConditionalGeneration.from_pretrained(llm_name)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.embed = self.llm.encoder.embed_tokens
        elif 't5' in llm_name:
            self.llm = T5ForConditionalGeneration.from_pretrained(llm_name)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.embed = self.llm.encoder.embed_tokens
        elif 'Llama' in llm_name:
            self.llm = LlamaForCausalLM.from_pretrained(
            llm_name
        )
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
            self.embed = self.llm.model.embed_tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token


        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        #self.spatio_proj = nn.Linear(self.input_size, self.inter_hidden, bias=False)

        #self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.llm.config.hidden_size)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.inter_hidden, self.llm.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        ) # 0906

        self.lip_backbone =  LipFeatureExtractor()
        self.lip_proj = build_vision_projector('linear', self.lip_backbone.feat_dim, self.inter_hidden)
        self.txt_proj = build_vision_projector('linear', self.llm.config.hidden_size, self.llm.config.hidden_size)

        if fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(inter_hidden * 2, inter_hidden),
                nn.Sigmoid()
            )
        elif fusion_mode == "qformer":
            self.video_qformer = QFormer(d_model=inter_hidden, out_model=self.llm.config.hidden_size, n_queries=num_queries)
            
            
        if fusion_mode == "residual_attention":
            self.ln1 = nn.LayerNorm(inter_hidden)
            self.ln2 = nn.LayerNorm(inter_hidden)
            self.ff = nn.Sequential(
                nn.Linear(inter_hidden, inter_hidden),
                nn.GELU(),
                nn.Linear(inter_hidden, inter_hidden)
                )
        elif fusion_mode == "concat":
            self.proj = nn.Linear(inter_hidden * 2, inter_hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inter_hidden,
            num_heads=4,
            batch_first=True
        )
        self.vis_token = nn.Parameter(torch.randn(1, 1, self.llm.config.hidden_size))  # [1, 1, D]
        self.prompt_token = nn.Parameter(torch.randn(1, 1, self.llm.config.hidden_size))

        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        if use_lora:
            self._apply_lora()

        self.use_align = use_align
        self.use_sign = use_sign
        self.use_spatial = use_spatial
        self.max_frame = 128
        self.overlap = overlap
        self.use_lip = use_lip
        self.use_vlign = use_vlign
        self.prompt_pos = prompt_pos
        self.num_queries = num_queries

        # Alignment scale
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))###wwf
        #self.logit_scale = nn.Parameter(torch.tensor(1.0))
        for name, param in self.llm.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def _apply_lora(self):
        if 't5' in self.llm_name:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "q", "k", "v", "o",  # 所有 attention 分支
                    "wi", "wo"  # Feed-forward 部分（DenseReluDense）
                ],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        else:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )

        self.llm = get_peft_model(self.llm, config)
        print("LoRA adapter applied to LLM model.")

    def _get_visual_outputs(self, spatial_feat, lip_video, num_frames, lip_length):
        spatial_out = self.spatio_proj(spatial_feat)
        lip_feat = self.lip_backbone(lip_video)#[batch, 768]
        lip_out = self.lip_proj(lip_feat)#[batch, 768]

        spatial_mask = create_mask(seq_lengths=num_frames, device=spatial_feat.device)
        lip_mask = create_mask(seq_lengths=lip_length, device=lip_feat.device)

        bs = spatial_out.shape[0]
        spatial_length = spatial_mask.sum(1)

        if self.use_spatial and self.use_lip:
            new_path = spatial_length

            if self.fusion_mode == "attention":
                # spatial_out, lip_out: [B, T, C]
                final_feat, _ = self.cross_attn(query=spatial_out, key=lip_out, value=lip_out, key_padding_mask=~lip_mask.bool())  # [B, T, C]
            elif self.fusion_mode == "residual_attention":
                # Cross attention + residual + norm + FFN
                attn_out, _ = self.cross_attn(query=spatial_out, key=lip_out, value=lip_out, key_padding_mask=~lip_mask.bool())
                x = self.ln1(attn_out + spatial_out)
                final_feat = self.ln2(self.ff(x) + x)
            elif self.fusion_mode == "gated":
                concat = torch.cat([spatial_out, lip_out], dim=-1)
                gate = self.gate(concat)
                final_feat = gate * spatial_out + (1 - gate) * lip_out

            elif self.fusion_mode == "qformer":
                final_feat = self.video_qformer(spatial_out, lip_out)
                B, Nq, D = final_feat.size()
                visual_masks = torch.ones(B, Nq, dtype=torch.long, device=final_feat.device)
                return final_feat, visual_masks, spatial_out, lip_out

            elif self.fusion_mode == "concat":
                concat = torch.cat([spatial_out, lip_out], dim=-1)  # 拼接特征维度
                final_feat = self.proj(concat)  # 映射回 hidden_dim

            else:
                raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

            # print(f"final_feat {final_feat.shape}")
            # Apply temporal encoder
            feat = final_feat.permute(0, 2, 1)  # (B, C, T)

            kernel_size = self.temporal_encoder.temporal_conv[0].kernel_size[0]
            if feat.size(-1) < kernel_size:
                pad_len = kernel_size - feat.size(-1)
                feat = F.pad(feat, (0, pad_len))  # 在时间维（最后一维）右侧补零

            visual_conv_outputs = self.temporal_encoder(
                feat, torch.tensor(new_path.tolist(), device=spatial_out.device)
            )

            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=spatial_out.device
            )
            '''
            print(f"temporal_outputs {temporal_outputs.shape}")
            B = lip_out.size(0)
            queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, C]
            visual_outputs, _ = self.cross_attn(query=queries, key=temporal_outputs, value=temporal_outputs,
                                            key_padding_mask=~temporal_masks.bool())
            visual_masks = torch.ones(B, self.num_queries, device=lip_out.device)

            print(f"visual_outputs {visual_outputs.shape}")
            print(f"visual_masks {visual_masks.shape}")'''

        elif self.use_spatial:
            spatial_conv_outputs = self.temporal_encoder(
                spatial_out.permute(0, 2, 1), torch.tensor(num_frames, device=spatial_feat.device)
            )
            visual_outputs = spatial_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=spatial_out.device
            )
        else:
            visual_outputs = lip_out
            visual_masks = lip_mask

        final_feats = self.fusion_proj(visual_outputs)
        #print(f"final_feats {final_feats.shape}")
        #
        #print(f"final_feats {final_feats.shape}")
        return final_feats, visual_masks, spatial_out, lip_out


    def _prepare_joint_inputs(self, visual_feats, video_mask, texts, context=None):
        bs = visual_feats.shape[0]

        prompts = [f'{texts}'] * bs
        if context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, context)]

        input_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )


        prompt_embeds = self.embed(input_tokens.input_ids.to(visual_feats.device))
        prompt_mask = input_tokens.attention_mask.to(visual_feats.device)
        #print(f"prompt_embeds: {prompt_embeds.shape}")
        #print(f"visual_feats: {visual_feats.shape}")

        # Get lengths for visual and prompt sequences
        visual_lengths = video_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths.to(visual_feats.device)

        # Expand special tokens
        vis_token = self.vis_token.expand(bs, -1, -1).to(visual_feats.device)  # [B, 1, D]
        prompt_token = self.prompt_token.expand(bs, -1, -1).to(visual_feats.device)  # [B, 1, D]

        joint_outputs = []
        joint_masks = []
        joint_lengths = []
        for i in range(bs):
            # vis_out = visual_feats[i, :visual_lengths[i], :]
            # new
            vis_out = visual_feats[i, :visual_lengths[i].item(), :]
            # prompt_out = prompt_embeds[i, :prompt_lengths[i], :]
            # new
            prompt_out = prompt_embeds[i, :prompt_lengths[i].item(), :]
            if self.prompt_pos == 1:
                concat_sample = torch.cat([vis_token[i], vis_out, prompt_token[i], prompt_out], dim=0)
            else:
                concat_sample = torch.cat((prompt_out, vis_out), dim=0)
            joint_outputs.append(concat_sample)
            mask = torch.cat([
                torch.ones(1, device=visual_feats.device),
                video_mask[i, :visual_lengths[i]],
                torch.ones(1, device=visual_feats.device),
                prompt_mask[i, :prompt_lengths[i]]
            ])
            joint_masks.append(mask)
            joint_lengths.append(concat_sample.size(0))

        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        #joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=visual_feats.device)
        joint_masks = pad_sequence(joint_masks, batch_first=True)
        joint_lengths = torch.tensor(joint_lengths, device=visual_feats.device)

        return joint_outputs, joint_masks, joint_lengths

    def _prepare_labels(self, text_ids):
        targets = text_ids.masked_fill(
            text_ids == self.tokenizer.pad_token_id, -100
        )
        return targets

    def build_llama_inputs_and_labels(self, inputs_embeds, attention_mask, joint_lengths, target_ids):
        """
        构造 LLaMA2 所需的 inputs_embeds, attention_mask 和 labels，支持 prompt + video_embeds + target 拼接训练。
        """
        # Step 1: 构造 target token 的 embedding
        target_embeds = self.embed(target_ids)  # [B, T, D]
        B, T, D = target_embeds.shape
        padded_input_embeds = F.pad(inputs_embeds, (0, 0, 0, T), value=0)  # [B, L+T, D]
        padded_input_embeds[:, -T:, :] = target_embeds

        # Step 2: 构造 attention mask
        new_attention_mask = F.pad(attention_mask, (0, T), value=0)  # [B, L+T]
        for i in range(B):
            new_attention_mask[i, joint_lengths[i]:joint_lengths[i] + T] = 1

        # Step 3: 构造 labels：前面部分设为 -100，后面是 target_ids
        labels = torch.full(
            (B, padded_input_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=target_ids.device
        )
        for i in range(B):
            labels[i, joint_lengths[i]:joint_lengths[i] + T] = target_ids[i]

        return padded_input_embeds, new_attention_mask, labels

    def get_visual_tokens(self, z_sign, tokenizer, label, top_k=1):
        """
        Use only tokens in `text_corpus` to build a smaller E_llm for nearest neighbor match.
        """
        # 1. 构建子词表
        full_embedding = self.llm.get_input_embeddings().weight.detach()
        print(f"label {label.shape}")
        token_ids = label.tolist()
        flat_token_ids = sum(token_ids, [])

        unique_ids = sorted(set(flat_token_ids))

        # 2. 提取 embedding 子集
        e_subset = full_embedding[unique_ids]  # [V', D]

        # 3. 计算最近邻
        z_flat = z_sign.view(-1, z_sign.shape[-1])
        z_norm = F.normalize(z_flat, dim=-1)
        e_norm = F.normalize(e_subset, dim=-1)
        sim = torch.matmul(z_norm, e_norm.T)  # [N, V']

        topk = torch.topk(sim, k=top_k, dim=-1)
        token_ids_k = topk.indices
        true_token_ids = [[unique_ids[i.item()] for i in row] for row in token_ids_k]

        # 4. decode
        if top_k == 1:
            tokens = [tokenizer.decode([i[0]]) for i in true_token_ids]
        else:
            tokens = [[tokenizer.decode([i]) for i in row] for row in true_token_ids]

        return tokens

    def forward(self, spatial_feat, lip_video, num_frames, lip_length, target_id, target_mask, context=None):
        visual_out, visual_mask, sign_out, lip_out = self._get_visual_outputs(spatial_feat, lip_video, num_frames, lip_length)

        txt_out = self.embed(target_id)
        #wwf
        #txt_out = self.txt_proj(text_embeds)

        input_embeds, attention_mask, lengths = self._prepare_joint_inputs(visual_out, visual_mask, self.prompt, context)
        #print(f"input_embeds {input_embeds.shape}")
        #print(f"attention_mask {attention_mask.shape}")

        if 't5' in self.llm_name:
            targets = self._prepare_labels(target_id)
        else:
            input_embeds, attention_mask, targets = self.build_llama_inputs_and_labels(input_embeds, attention_mask, lengths, target_id)

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_attention_mask=target_mask,
            labels=targets,
            output_hidden_states = True,
            return_dict = True
        )
        llm_loss = outputs.loss

        if self.use_vlign and self.use_align:
            #sign_llm = self.fusion_proj(sign_out)
            #lip_llm = self.fusion_proj(lip_out)
            cross_loss = self.visual_textual_align(txt_out, visual_out)
            vis_loss = self.lip_sign_align(sign_out, lip_out, self.use_frame)

            #lip_loss = self.visual_textual_align(target_id, lip_llm)

            total_loss = llm_loss + self.alpha * cross_loss + self.beta * vis_loss

        # Lip ↔ Sign 对比损失（模态内对齐）
        elif self.use_vlign:
            vis_loss = self.lip_sign_align(sign_out, lip_out)
            total_loss = llm_loss + self.beta * vis_loss  # 使用 beta 控制权重

        # Visual ↔ Text 对比损失（跨模态对齐）
        elif self.use_align and target_id is not None:
            con_loss = self.visual_textual_align(target_id, visual_out)
            total_loss = llm_loss + self.alpha * con_loss  # 使用 alpha 控制权重
        else:
            total_loss = llm_loss

        return total_loss

    def generate(self, spatial_feat, lip_video, num_frames, lip_length, context, target_id=None):

        visual_out, visual_mask, sign_out, lip_out = self._get_visual_outputs(spatial_feat, lip_video, num_frames, lip_length)
        #print(f"lip_out {lip_out.shape}")
        #print(f"sign_out {sign_out.shape}")

        input_embeds, attention_mask, lengths = self._prepare_joint_inputs(visual_out, visual_mask, self.prompt, context)
        #print(f"input_embeds {input_embeds.shape}")
        #print(f"attention_mask {attention_mask.shape}")
        assert input_embeds.shape[:2] == attention_mask.shape

        #print(f"input_embeds {input_embeds[0]}")


        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=5,
            early_stopping=True,
            #top_p=0.9,
            do_sample=False,
        )
        '''
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=5,
            #early_stopping=True,
            top_p=0.9,
            do_sample=True,
        )
        '''
        out_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outs = [out.lower() for out in out_text]

        return outs

    def self_align(self, self_out):

        self_pool = self_out.mean(1)
        self_pool = F.normalize(self_pool, dim=-1)
        logits_per_self = torch.matmul(self_pool, self_pool.T) * self.logit_scale.exp()
        return clip_loss(logits_per_self)

    def visual_textual_align(self, text_out, visual_outputs):
        if text_out is None:
            return torch.tensor(0.0, device=visual_outputs.device)

        image_embeds = visual_outputs.mean(1)
        text_embeds = text_out.mean(1)

        #text_embeds = self.txt_proj(text_out[:, -1, :])

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * self.logit_scale.exp()
        return clip_loss(logits_per_text)

    def lip_sign_frame_align(self, sign_out, lip_out, gap=5, temperature=0.07):
        """
        Frame-level Lip-Sign 对齐 (排除近邻帧作为负例)
        Args:
            sign_out: [B, T, D]
            lip_out:  [B, T, D]
            gap: 正负例分界，|t-k| <= gap 的帧不作为负例
        """
        B, T, D = sign_out.shape

        # 归一化
        sign_norm = F.normalize(sign_out, dim=-1)  # [B, T, D]
        lip_norm = F.normalize(lip_out, dim=-1)

        # soft alignment: sign[t] -> lip 对应帧的加权和
        attn = torch.matmul(sign_norm, lip_norm.transpose(1, 2)) / (D ** 0.5)  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        lip_aligned = torch.bmm(attn, lip_norm)  # [B, T, D]

        # flatten
        N = B * T
        sign_flat = sign_norm.reshape(N, D)  # [N, D]
        lip_pos_flat = lip_aligned.reshape(N, D)  # 正例 [N, D]
        lip_all_flat = lip_norm.reshape(N, D)  # 所有 lip 帧 [N, D]

        # 全相似度矩阵
        logits_all = torch.matmul(sign_flat, lip_all_flat.T) / temperature  # [N, N]

        # 正例 logits (soft aligned)
        pos_logits = torch.sum(sign_flat * lip_pos_flat, dim=-1, keepdim=True) / temperature  # [N, 1]

        # 构造 mask：同 batch 内，排除当前样本的近邻帧 (|t-k| <= gap)
        mask = torch.ones(N, N, device=sign_out.device)
        for b in range(B):
            for t in range(T):
                idx = b * T + t
                for k in range(max(0, t - gap), min(T, t + gap + 1)):  # 局部窗口
                    mask[idx, b * T + k] = 0

        # 应用 mask，屏蔽掉近邻帧
        neg_logits = logits_all.masked_fill(mask == 0, -1e9)  # [N, N]

        # 拼接正例 + 负例
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1+N]
        labels = torch.zeros(N, dtype=torch.long, device=sign_out.device)  # 正例在 index=0

        loss = F.cross_entropy(logits, labels)
        return loss

    def lip_sign_align(self, lip_out, sign_out, use_frame=False, frame_gap=5):
        """
        Lip ↔ Sign 对齐
        支持 sentence-level + frame-level 混合
        """
        # ===== sentence-level ====
        lip_embeds = lip_out.mean(1)  # [B, D]
        sign_embeds = sign_out.mean(1)

        lip_embeds = F.normalize(lip_embeds, dim=-1)
        sign_embeds = F.normalize(sign_embeds, dim=-1)

        logits_per_lip = torch.matmul(lip_embeds, sign_embeds.T) * self.logit_scale.exp()
        sent_loss = clip_loss(logits_per_lip)

        if use_frame:
            frame_loss = self.lip_sign_frame_align(sign_out, lip_out, gap=frame_gap)
            return frame_loss+sent_loss
        return sent_loss



