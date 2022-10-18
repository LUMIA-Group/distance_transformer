# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor


@dataclass
class MM2VecCriterionConfig(FairseqDataclass):
    infonce: bool = field(
        default=False,
        metadata={
            "help": "if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)"
        },
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    mse_only: bool = field(
        default=False,
        metadata={"help" : "calculate mse loss only in stage 1"}
    )
    cpc_only: bool = field(
        default=False,
        metadata={"help": "calculate cpc loss only in stage 1"}
    )
    add_mse: bool = field(
        default=False,
        metadata={"help": "add mse loss in stage 2"}
    )
    pre_gelu_mse: bool = field(
        default=False,
        metadata={"help":"mse after or before gelu"}
    )


@register_criterion("mm2vec", dataclass=MM2VecCriterionConfig)
class MM2vecCriterion(FairseqCriterion):
    def __init__(self, task, cfg: MM2VecCriterionConfig):
        super().__init__(task)
        self.infonce = cfg.infonce
        self.loss_weights = cfg.loss_weights
        self.log_keys = [] if cfg.log_keys is None else cfg.log_keys
        self.mse_only = cfg.mse_only
        self.cpc_only = cfg.cpc_only
        self.add_mse = cfg.add_mse
        self.pre_gelu_mse = cfg.pre_gelu_mse

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print(self.log_keys)
        net_output = model(sample["net_input"]['audio_source'], sample["net_input"]['visual_source'])
        logits_audio,logits_visual = model.get_logits(net_output)
        logits_audio = logits_audio.float()
        logits_visual = logits_visual.float()
        target_audio,target_visual = model.get_targets(sample, net_output)
        self.xla = is_xla_tensor(logits_audio) and is_xla_tensor(logits_visual)
        self.model_stage = net_output["stage"]
        # XXX: handle weights on xla.
        weights = None
        # ignore
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []
        logging_output = {}

        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            if self.model_stage == 1:
                # CPC_VISUAL_LOSS + MSE_LOSS
                CPC_visual_loss = F.cross_entropy(logits_visual, target_visual)
                PRE_MSE_loss = F.mse_loss(net_output['pre_gelu_visual'], net_output['pre_gelu_audio'])
                POST_MSE_loss = F.mse_loss(net_output["post_gelu_visual"], net_output["post_gelu_audio"])
                logging_output["PRE_MSE_loss"] = PRE_MSE_loss
                logging_output["POST_MSE_loss"] = POST_MSE_loss
                logging_output["CPC_visual_loss"] = CPC_visual_loss
                logging_output["CPC_audio_loss"] = F.cross_entropy(logits_audio, target_audio)
                if self.mse_only and not self.cpc_only:
                    if self.pre_gelu_mse:
                        loss = PRE_MSE_loss
                    else:
                        loss = POST_MSE_loss
                if not self.mse_only and self.cpc_only:
                    loss = CPC_visual_loss
                if not self.mse_only and not self.cpc_only:
                    if self.pre_gelu_mse:
                        loss = CPC_visual_loss + PRE_MSE_loss
                    else:
                        loss = CPC_visual_loss + POST_MSE_loss

            elif self.model_stage ==2:
                # CPC_VISUAL_LOSS +CPC_AUDIO_LOSS
                CPC_visual_loss = F.cross_entropy(logits_visual, target_visual)
                CPC_audio_loss = F.cross_entropy(logits_audio, target_audio)
                PRE_MSE_loss = F.mse_loss(net_output['pre_gelu_visual'], net_output['pre_gelu_audio'])
                POST_MSE_loss = F.mse_loss(net_output["post_gelu_visual"], net_output["post_gelu_audio"])
                logging_output["CPC_visual_loss"] = CPC_visual_loss
                logging_output["CPC_audio_loss"] = CPC_audio_loss
                logging_output["PRE_MSE_loss"] = PRE_MSE_loss
                logging_output["POST_MSE_loss"] = POST_MSE_loss
                if self.add_mse:
                    if self.pre_gelu_mse:
                        loss = CPC_visual_loss + CPC_audio_loss + PRE_MSE_loss
                    else:
                        loss = CPC_visual_loss + CPC_audio_loss + POST_MSE_loss
                else:
                    loss = CPC_visual_loss + CPC_audio_loss

        else:
            # ignore
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )
        # ignore
        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample['net_input']['mask_indices']
                    .transpose(0, 1)  # logits are transposed in `model.get_logits`
                    .reshape(logits.size(0))
            )
            loss = (loss * mi).sum() if reduce else (loss * mi)

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target_audio.numel() if self.infonce else target_audio.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float()
                    loss += p
                    losses.append(p)

        logging_output["loss"] = loss.item() if (reduce and not self.xla) else loss.detach()
        logging_output["ntokens"] = sample_size
        logging_output["nsentences"] = sample["id"].numel()
        logging_output["sample_size"] = sample_size
        logging_output["code_perplexity"] = net_output["code_perplexity"]
        logging_output["num_vars"] = net_output["num_vars"]

        # for lk in self.log_keys:
        #     # Only store "logits" and "target" for computing MAP and MAUC
        #     # during validation
        #     if lk == "logits":
        #         if not self.training:
        #             logging_output["logits"] = logits.cpu().numpy()
        #     elif lk == "target":
        #         if not self.training:
        #             # If the targets have been mixed with the predictions of
        #             # teacher models, find the original targets
        #             if hasattr(model, "get_original_targets"):
        #                 original_target = model.get_original_targets(sample, net_output)
        #             else:
        #                 original_target = target
        #             logging_output["target"] = original_target.cpu().numpy()
        #     elif lk in net_output:
        #         value = net_output[lk]
        #         if not is_xla_tensor(value):
        #             value = float(value)
        #         logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item() if not self.xla else l.detach()

        # if self.infonce:
        #     with torch.no_grad():
        #         if logits.numel() == 0:
        #             corr = 0
        #             count = 0
        #         else:
        #             assert logits.dim() > 1, logits.shape
        #             max = logits.argmax(-1) == 0
        #             min = logits.argmin(-1) == 0
        #             if is_xla_tensor(logits):
        #                 max, min = max * mi, min * mi
        #                 both = max & min
        #                 corr = max.long().sum() - both.long().sum()
        #                 count = mi.sum()
        #             else:
        #                 both = max & min
        #                 corr = max.long().sum().item() - both.long().sum().item()
        #                 count = float(max.numel())
        #
        #         logging_output["correct"] = corr
        #         logging_output["count"] = count

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum)
        metrics.log_scalar("LOSS/CPC VISUAL LOSS", float(logging_outputs[0]["CPC_visual_loss"]))
        metrics.log_scalar("LOSS/CPC AUDIO LOSS", float(logging_outputs[0]["CPC_audio_loss"]))
        metrics.log_scalar("LOSS/PRE MSE LOSS", float(logging_outputs[0]["PRE_MSE_loss"]))
        metrics.log_scalar("LOSS/POST MSE LOSS", float(logging_outputs[0]["POST_MSE_loss"]))
        metrics.log_scalar("LOSS/DIVERSITY LOSS VISUAL", float(logging_outputs[0]["loss_1"]))
        if "loss_2" in logging_outputs[0]:
            metrics.log_scalar("LOSS/DIVERSITY LOSS AUDIO", float(logging_outputs[0]["loss_2"]))
        if "code_perplexity" in logging_outputs[0]:
                metrics.log_scalar("CODE_PPL/VISUAL CODE PPL", float(logging_outputs[0]["code_perplexity"][0]))
                metrics.log_scalar("CODE_PPL/AUDIO CODE PPL", float(logging_outputs[0]["code_perplexity"][1]))
        metrics.log_scalar("sample_size",sample_size)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        # correct = sum(log.get("correct", 0) for log in logging_outputs)
        # metrics.log_scalar("_correct", correct)
        #
        # total = sum(log.get("count", 0) for log in logging_outputs)
        # metrics.log_scalar("_total", total)
        #
        # if total > 0:
        #     metrics.log_derived(
        #         "accuracy",
        #         lambda meters: safe_round(
        #             meters["_correct"].sum / meters["_total"].sum, 5
        #         )
        #         if meters["_total"].sum > 0
        #         else float("nan"),
        #     )
        #
        # builtin_keys = {
        #     "loss",
        #     "ntokens",
        #     "nsentences",
        #     "sample_size",
        #     "correct",
        #     "count",
        # }
        #
        # for k in logging_outputs[0]:
        #     if k not in builtin_keys:
        #         val = sum(log.get(k, 0) for log in logging_outputs)
        #         if k.startswith("loss"):
        #             metrics.log_scalar(
        #                 k, val / (sample_size or 1) / math.log(2), sample_size, round=3
        #             )
        #         else:
        #             metrics.log_scalar(k, val / len(logging_outputs), round=3)

    # FIXME: revert when gather based xla reduction is implemented
    # @staticmethod
    # def logging_outputs_can_be_summed() -> bool:
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # XXX: Gather based reduction not implemented for xla yet.
        # So we fall to sum based reduction for xla.
        return self.xla
