# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
from torchvision.models.resnet import resnet50


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class MM2VecConfig(FairseqDataclass):
    model_stage: int = field(
        default=1,
        metadata={"help": "model_stage=1 for training visual feature extractor only,"
                          "model_stage=2 for pretrain on all subnet"
                          "model_stage=? for fine-tune"},
    )

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    audio_conv_feature_layers: str = field(
        default="[(512, 10, 5, 0)] + [(512, 3, 2, 0)] * 4 + [(512, 2, 2, 0)] + [(512, 2, 2, 0)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )

    # Visual Part
    visual_conv_feature_layers: str = field(
        default="[(512, 11, 1, 5)] * 3 + [(1024, 11, 1, 5)]",
        metadata={
            "help": "string describing visual-subnet convolutional feature extraction layers in form of a python list that contains "
                    "[(dim, kernel_size, stride, padding), ...]"
        },
    )
    visual_input_dim: int = field(
        default=112,
        metadata={"help": "number of dims of visual pictures"},
    )
    visual_encoder_dim: int = field(
        default=2048,
        metadata={"help": "number of dims after MoCo"},
    )
    projection_dim: int = field(
        default=512,
        metadata={"help": "output dimension of projection head"},
    )

    # checkpoint part
    m2v_path : str = field(
        default="./checkpoints-mm-2/",
        metadata={
            "help": "path to mm2vec stage 1 last model or stage 2 process model"
        },
    )
    # aggregation part
    audio_weight: float = field(
        default=0.5,
        metadata={
            "help":"weight for audio_features"
        }
    )
    visual_weight: float = field(
        default=0.5,
        metadata={
            "help":"weight for audio_features"
        }
    )
    remove_quantizer_weight: bool = field(
        default=False,
        metadata={
            "help": "remove quantizer pretrain params"
        }
    )
    unfreeze_quantizer_weight:bool = field(
        default=False,
        metadata={
            "help": "freeze quantizer pretrain params"
        }
    )
    # MoCo
    MoCo_replace:bool = field(
        default=False,
        metadata={"help":"replace first conv2d in MoCo with conv3d"}
    )

@register_model("mm2vec", dataclass=MM2VecConfig)
class MM2VecModel(BaseFairseqModel):
    def __init__(self, cfg: MM2VecConfig):
        super().__init__()
        self.cfg = cfg

        audio_feature_enc_layers = eval(cfg.audio_conv_feature_layers)
        visual_feature_enc_layers = eval(cfg.visual_conv_feature_layers)
        self.audio_embed_dim = audio_feature_enc_layers[-1][0]      # 512
        self.visual_embed_dim = visual_feature_enc_layers[-1][0]    # 1024
        self.projection_dim = cfg.projection_dim                    # 512
        self.audio_feature_extractor = ConvFeatureExtractionModel(
            conv_layers=audio_feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            input_dim=1,
        )

        self.visual_input_dim = cfg.visual_input_dim    # 112
        self.MoCo_replace = cfg.MoCo_replace
        self.MoCo_extractor = MoCo(replace=self.MoCo_replace)
        self.visual_encoder_dim = cfg.visual_encoder_dim        # 2048

        self.visual_feature_extractor = ConvFeatureExtractionModel(
            conv_layers=visual_feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            input_dim=2048,
        )

        self.post_extract_proj = (
            # 512 -> 768
            nn.Linear(self.audio_embed_dim, cfg.encoder_embed_dim)
            if self.audio_embed_dim != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.projection_head = nn.Sequential(
            # 512 -> 512
            nn.Linear(int(self.visual_embed_dim / 2), int(self.visual_embed_dim / 2), bias=False),
            nn.ReLU(),
            # 512 -> 768
            nn.Linear(int(self.visual_embed_dim / 2), cfg.encoder_embed_dim, bias=False),
        )

        """ mask part """
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        """ mask part """

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.audio_embed_dim,   # 512
                num_vars=cfg.latent_vars,   # 320
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,   # 2
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        # if cfg.quantize_input:
        #     if cfg.same_quantizer and self.quantizer is not None:
        #         vq_dim = final_dim
        #         self.input_quantizer = self.quantizer
        #     else:
        #         vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
        #         self.input_quantizer = GumbelVectorQuantizer(
        #             dim=self.embed,
        #             num_vars=cfg.latent_vars,
        #             temp=cfg.latent_temp,
        #             groups=cfg.latent_groups,
        #             combine_groups=False,
        #             vq_dim=vq_dim,
        #             time_first=True,
        #             weight_proj_depth=cfg.quantizer_depth,
        #             weight_proj_factor=cfg.quantizer_factor,
        #         )
        #     self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.audio_embed_dim)
        self.visual_layer_norm = LayerNorm(int(self.visual_embed_dim / 2))

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.model_stage = cfg.model_stage
        self.audio_weight = cfg.audio_weight
        self.visual_weight = cfg.visual_weight

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: MM2VecConfig, task=None):
        """Build a new model instance."""
        model = cls(cfg)
        if cfg.model_stage == 1:
            model_dict = model.state_dict()
            wav2vec_dict = {k.replace('feature', 'audio_feature'): v for k, v in
                            torch.load('../pretrain/wav2vec_small.pt')["model"].items()}
            moco_dict = {k.replace('module.encoder_q', 'MoCo_extractor.encoder'): v for k, v in
                         torch.load('../pretrain/moco_v2_800ep_pretrain.pth.tar')["state_dict"].items()}
            if cfg.remove_quantizer_weight:
                popKeys = ['quantizer.vars', 'quantizer.weight_proj.weight', 'quantizer.weight_proj.bias']
                for k in popKeys:
                    wav2vec_dict.pop(k)
            popKeys = ['MoCo_extractor.encoder.fc.0.bias', 'MoCo_extractor.encoder.fc.2.bias',
                       'MoCo_extractor.encoder.fc.0.weight', 'MoCo_extractor.encoder.fc.2.weight']
            if cfg.MoCo_replace:
                popKeys.append('MoCo_extractor.encoder.conv1.weight')
            for k in popKeys:
                moco_dict.pop(k)
            model_dict.update(wav2vec_dict)
            model_dict.update(moco_dict)
            model.load_state_dict(model_dict)
            popKeys = ['quantizer.vars', 'quantizer.weight_proj.weight', 'quantizer.weight_proj.bias']
            for name, param in model.named_parameters():
                # print(name)
                if name in wav2vec_dict.keys() or name in moco_dict.keys():
                    param.requires_grad = False
                    if name in popKeys and cfg.unfreeze_quantizer_weight:
                        param.requires_grad = True
        elif cfg.model_stage == 2:
            model_dict = model.state_dict()
            checkpoint_path = os.path.join(cfg.m2v_path, 'checkpoint_last.pt')
            checkpoints_dict = torch.load(checkpoint_path)['model']
            model_dict.update(checkpoints_dict)
            model.load_state_dict(model_dict)
        else:
            return model
        print('num_total_param: {},num_trainable_param: {},num_freezed_param: {}'.format(
            sum([params.numel() for params in model.parameters()]),
            sum([params.numel() for params in model.parameters() if params.requires_grad]),
            sum([params.numel() for params in model.parameters() if not params.requires_grad])))
        return model

    def apply_mask(
        self,
        x_audio,
        x_visual,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x_audio.shape

        # FIXME INFERENCE
        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x_audio.device)
            x_audio = index_put(x_audio, mask_indices, self.mask_emb)
            x_visual = index_put(x_visual, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        # FIXME INFERENCE
        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x_audio.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x_audio = index_put(x_audio, mask_channel_indices, 0)
            x_visual = index_put(x_visual, mask_channel_indices, 0)

        return x_audio, x_visual, mask_indices

    def sample_negatives(self, y_audio, y_visual, num, padding_count=None):

        #ignore
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y_audio.shape
        y_audio = y_audio.view(-1, fsz)  # BTC => (BxT)C
        y_visual = y_visual.view(-1, fsz)

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs_audio = y_audio[neg_idxs.view(-1)]
        negs_audio = negs_audio.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC

        negs_visual = y_visual[neg_idxs.view(-1)]
        negs_visual = negs_visual.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs_audio, negs_visual, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / self.logit_temp

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.audio_conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def compute_visual_length(self,visual_source):
        visual_length = list()
        max_visual_length = -1
        for i in range(len(visual_source)):
            length = int(visual_source[i].size(1) / self.visual_input_dim)
            if length > max_visual_length:
                max_visual_length = length
            visual_length.append(length)
        return max_visual_length,visual_length

    def visual_padding(self,visual_features,visual_length,max_visual_length):
        visual_source_new = torch.tensor([], dtype=visual_features.dtype, device=visual_features.device)
        start = 0
        # 根据visual length数组切分MoCo的输出结果，并padding到最长
        visual_source_len = max_visual_length
        for l in visual_length:
            visual_source_new = torch.cat((visual_source_new, torch.cat(
                (visual_features[start:start + l],
                 torch.zeros((visual_source_len - l, 3,112,112), dtype=visual_features.dtype,
                             device=visual_features.device)))))
        return visual_source_new

    def forward(
        self,
        audio_source,
        visual_source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        """
        先只管cropping的训练模式，stage1 stage2都是对应好了的 visual 和 audio 长度 
        batch内不同sample的visual length或者 audio length都是一样的
        不需要算长度序列
        inference：dataset的pad参数被设置 需要传入padding mask的时候，audio的source才是padding了的
        这个时候才需要记录visual length，并在过完moco之后padding
        """
        result = {}
        # FIXME INFERENCE
        if padding_mask is not None:
            # compute visual length
            max_visual_length, visual_length = self.compute_visual_length(visual_source)

            visual_source = torch.cat(visual_source,1)
            visual_source = torch.split(visual_source, self.visual_input_dim, 1)
            visual_source = torch.cat(visual_source)
            visual_source = visual_source.view(-1, self.visual_input_dim, self.visual_input_dim)
            visual_source = visual_source.unsqueeze(1).repeat(1, 3, 1, 1)
            if self.MoCo_replace:
                visual_source = self.visual_padding(visual_source,visual_length,max_visual_length)
                visual_source = visual_source.view(len(visual_length),max_visual_length,3,112,112)
                visual_source = visual_source.transpose(1,2)

        else:
            """
            cropping训练，batch内的visual input长度一样
            """
            visual_batch_size = len(visual_source)
            max_visual_length = int(visual_source[0].size(1)/112)
            visual_source = torch.stack(visual_source)
            visual_source = torch.split(visual_source, self.visual_input_dim, 1)
            visual_source = torch.cat(visual_source)
            visual_source = visual_source.view(-1, self.visual_input_dim, self.visual_input_dim)
            visual_source = visual_source.unsqueeze(1).repeat(1, 3, 1, 1)
            if self.MoCo_replace:
                visual_source = visual_source.view(visual_batch_size, max_visual_length, 3, self.visual_input_dim, self.visual_input_dim)
                visual_source = visual_source.transpose(1, 2)

        """MoCo input dim:[n_frames,3,112,112]"""
        visual_features = self.MoCo_extractor(visual_source)
        visual_features = visual_features.view(-1,max_visual_length,self.visual_encoder_dim)
        visual_features = visual_features.transpose(1,2)

        """
        长度问题到这里应该就结束了，后面不管是padding还是cropping都是align好了的
        """

        if self.feature_grad_mult > 0:
            # audio: (bsz*sample_length) --> (bsz * feature_dim * frames)
            # visual: (bsz*feature_dim * frames) --> (bsz * feature_dim_new * frames)
            af_beforeGELU, audio_features = self.audio_feature_extractor(audio_source)
            vf_beforeGELU, visual_features = self.visual_feature_extractor(visual_features)
            if self.feature_grad_mult != 1.0:
                audio_features = GradMultiply.apply(audio_features, self.feature_grad_mult)
                visual_features = GradMultiply.apply(visual_features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                af_beforeGELU, audio_features = self.audio_feature_extractor(audio_source)
                vf_beforeGELU, visual_features = self.visual_feature_extractor(visual_features)

        features_pen = 0 # penalty loss

        af_beforeGELU = af_beforeGELU.transpose(1,2)
        vf_beforeGELU = vf_beforeGELU.transpose(1,2)
        vf_beforeGELU = vf_beforeGELU.reshape(vf_beforeGELU.size(0), -1,int(vf_beforeGELU.size(2) / 2))
        vf_beforeGELU = vf_beforeGELU[:, :af_beforeGELU.size(1), :]
        af_beforeGELU = self.layer_norm(af_beforeGELU)
        vf_beforeGELU = self.visual_layer_norm(vf_beforeGELU)

        result["pre_gelu_audio"] = af_beforeGELU
        result["pre_gelu_visual"] = vf_beforeGELU

        # FIXME:做不做transpose和layer_norm对MSE的影响是啥?过不过GELU的MSE区别是啥?

        audio_features = audio_features.transpose(1, 2)
        visual_features = visual_features.transpose(1, 2)
        visual_features = visual_features.reshape(visual_features.size(0), -1, int(visual_features.size(2) / 2))
        visual_features = visual_features[:, :audio_features.size(1), :]
        audio_features = self.layer_norm(audio_features)         # 512维度上做的layernorm
        visual_features = self.visual_layer_norm(visual_features)

        result["post_gelu_audio"] = audio_features
        result["post_gelu_visual"] = visual_features

        unmasked_audio_features = audio_features.clone()
        unmasked_visual_features = visual_features.clone()

        # FIXME INFERENCE
        """sample维度的padding mask到frame维度的padding mask"""
        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                audio_features.shape[:2], dtype=audio_features.dtype, device=audio_features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        # 512 -> 768
        if self.post_extract_proj is not None:
            audio_features = self.post_extract_proj(audio_features)
            visual_features = self.post_extract_proj(visual_features)

        # if self.projection_head is not None:
        #     visual_features = self.projection_head(visual_features)

        result["features_pen"] = features_pen

        audio_features = self.dropout_input(audio_features)
        visual_features = self.dropout_input(visual_features)
        unmasked_audio_features = self.dropout_features(unmasked_audio_features)
        unmasked_visual_features = self.dropout_features(unmasked_visual_features)


        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        # if self.input_quantizer:
        #     q = self.input_quantizer(features, produce_targets=False)
        #     features = q["x"]
        #     num_vars = q["num_vars"]
        #     code_ppl = q["code_perplexity"]
        #     prob_ppl = q["prob_perplexity"]
        #     curr_temp = q["temp"]
        #     features = self.project_inp(features)

        if mask:
            # inference的时候不计算mask / compute mask indices and set (indices==True) position as self.mask_emb
            x_audio, x_visual, mask_indices = self.apply_mask(
                audio_features,
                visual_features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x_audio) and not is_xla_tensor(x_audio) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y_audio = unmasked_audio_features[mask_indices].view(
                    unmasked_audio_features.size(0), -1, unmasked_audio_features.size(-1)
                )
                y_visual = unmasked_visual_features[mask_indices].view(
                    unmasked_visual_features.size(0), -1, unmasked_visual_features.size(-1)
                )
            else:
                # ignore
                y_audio = unmasked_audio_features
                y_visual = unmasked_visual_features
        else:
            x_audio = audio_features
            x_visual = visual_features
            y_audio = unmasked_audio_features
            y_visual = unmasked_visual_features
            mask_indices = None

        """
        mask之后的过transformer
        stage 1:  两个模态分别过
        stage 2： 两个模态取平均后过
        """
        if self.model_stage == 1:
            """
                x_audio:Batch * n_frames(with mask_emb) * feature_dim(512)
                x_visual:Batch * n_frames(with mask_emb) * feature_dim(512)
                x_audio.shape == x_visual.shape
            """
            x_audio, layer_results_audio = self.encoder(x_audio, padding_mask=padding_mask, layer=layer)
            x_visual, layer_results_visual = self.encoder(x_visual, padding_mask=padding_mask, layer=layer)
        elif self.model_stage == 2:
            x_cat = (self.audio_weight * x_audio + self.visual_weight * x_visual)
            x_cat,layer_results_cat = self.encoder(x_cat, padding_mask=padding_mask, layer=layer)
        else:
            x_cat = (0.0 * x_audio + 1.0 * x_visual)
            x_cat, _ = self.encoder(x_cat, padding_mask=padding_mask, layer=layer)

        # FIXME INFERENCE
        if features_only:
            return {
                "x": x_cat,
                "padding_mask": padding_mask,
                "audio_features": unmasked_audio_features,
                "visual_features": unmasked_visual_features,
            }
        """
        inference时到这儿就结束了
        """
        if self.quantizer:
            q_visual = self.quantizer(y_visual, produce_targets=False)
            y_visual = q_visual["x"]
            q_audio = self.quantizer(y_audio, produce_targets=False)
            y_audio = q_audio["x"]
            if self.model_stage == 1:
                """
                只管visual这边的diversity loss
                """
                num_vars = q_visual["num_vars"]
                code_ppl = [q_visual["code_perplexity"], q_audio["code_perplexity"]]
                # 进入码本的比例 = code_ppl/(num_vars*num_latent_groups)
                # print("visual_num_vars:",num_vars)
                # print("audio_num_vars:", q_audio["num_vars"])
                # print("visual_code_ppl:", code_ppl)
                # print("audio_code_ppl:", q_audio["code_perplexity"])
                prob_ppl = q_visual["prob_perplexity"]
                curr_temp = q_visual["temp"]
            elif self.model_stage == 2:
                num_vars = q_visual["num_vars"]
                code_ppl = [q_visual["code_perplexity"], q_audio["code_perplexity"]]
                # print("num_vars_va:", num_vars)
                # print("code_ppl_va:", code_ppl)
                prob_ppl = [q_visual["prob_perplexity"], q_audio["prob_perplexity"]]
                curr_temp = [q_visual["temp"], q_audio["temp"]]
            y_audio = self.project_q(y_audio)
            y_visual = self.project_q(y_visual)

            # ignore
            if self.negatives_from_everywhere:
                # ignore
                neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                    "x"
                ]
                negs, _ = self.sample_negatives(
                    neg_cands,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs_audio,negs_visual, negs_indices = self.sample_negatives(
                    y_audio,
                    y_visual,
                    y_audio.size(1),
                    padding_count=padding_count,
                )

            # ignore
            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y_audio = self.project_q(y_audio)
            y_visual = self.project_q(y_visual)

            #ignore
            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x_audio) and not is_xla_tensor(x_visual) and self.model_stage == 1:
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x_audio = x_audio[mask_indices].view(x_audio.size(0), -1, x_audio.size(-1))
            x_visual = x_visual[mask_indices].view(x_visual.size(0), -1, x_visual.size(-1))
        elif not is_xla_tensor(x_cat) and self.model_stage == 2:
            x_cat = x_cat[mask_indices].view(x_cat.size(0), -1, x_cat.size(-1))

        # ignore
        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        if self.model_stage == 1:

            x_audio = self.final_proj(x_audio)
            x_audio = self.compute_preds(x_audio, y_audio, negs_audio)
            x_visual = self.final_proj(x_visual)
            x_visual = self.compute_preds(x_visual, y_visual, negs_visual)
            result["x_audio"] = x_audio
            result["x_visual"] = x_visual
            result["padding_mask"] = padding_mask

        elif self.model_stage == 2:
            x_cat = self.final_proj(x_cat)
            x_audio = self.compute_preds(x_cat, y_audio, negs_audio)
            x_visual = self.compute_preds(x_cat, y_visual, negs_visual)
            result["x_audio"] =x_audio
            result["x_visual"] = x_visual
            result["padding_mask"] = padding_mask

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp
        result["stage"] = self.model_stage
        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, audio_source, visual_source, padding_mask, mask=False, layer=None):
        res = self.forward(
            audio_source,visual_source, padding_mask, mask=mask, features_only=True, layer=layer
        )
        return res

    def get_logits(self, net_output):
        logits_audio = net_output["x_audio"]
        logits_visual = net_output["x_visual"]
        logits_audio = logits_audio.transpose(0, 2)
        logits_visual = logits_visual.transpose(0, 2)
        logits_audio = logits_audio.reshape(-1, logits_audio.size(-1))
        logits_visual = logits_visual.reshape(-1, logits_audio.size(-1))
        return logits_audio,logits_visual

    def get_targets(self, sample, net_output, expand_steps=True):
        x_audio = net_output["x_audio"]
        x_visual = net_output["x_visual"]
        return x_audio.new_zeros(x_audio.size(1) * x_audio.size(2), dtype=torch.long), x_visual.new_zeros(x_visual.size(1) * x_visual.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            if self.model_stage == 1:
                pen.append(
                    (net_output["num_vars"] - net_output["prob_perplexity"])
                    / net_output["num_vars"]
                )
            else:
                for i in range(2):
                    # visual audio
                    pen.append(
                        (net_output["num_vars"] - net_output["prob_perplexity"][i])
                        / net_output["num_vars"]
                    )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            input_dim=1,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                kernel_size,
                stride,
                padding,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = input_dim
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim, kernel_size, stride, padding) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    kernel_size,
                    stride,
                    padding,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            if conv == self.conv_layers[-1]:
                for name, module in conv.named_children():
                    if name =="2":
                        """
                            0 Conv1d
                            1 Dropout
                            2 GELU
                            2 means GELU()
                        """
                        before_GELU = x
                    x = module(x)
            else:
                x = conv(x)
        return before_GELU, x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn

class MoCo(nn.Module):

    def __init__(self, replace=False):
        super(MoCo, self).__init__()
        self.encoder = nn.Sequential()
        self.replace = replace
        for name, module in resnet50().named_children():
            """
            name:conv1
            name:bn1
            name:relu
            name:maxpool
            name:layer1
            name:layer2
            name:layer3
            name:layer4
            name:avgpool
            name:fc
            """
            if name == 'conv1':
                if self.replace:
                    module = nn.Conv3d(3,64,kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
                self.encoder.add_module(name, module)
            elif name != 'fc':
                self.encoder.add_module(name, module)
            # else:
            #     self.ResNet.append(nn.Linear(in_features=2048, out_features=128, bias=True))

    def forward(self, x):
        x = self.encoder.conv1(x)
        if self.replace:
            x = x.transpose(1,2)
            x = x.reshape(-1,x.size(2),x.size(3),x.size(4))
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)

        feature = torch.flatten(x, start_dim=1)
        return F.normalize(feature, dim=-1)