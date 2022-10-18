# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from .fairseq_dataset import FairseqDataset
from .data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
# from fairseq.data.audio.audio_utils import (
#     parse_path,
#     read_from_stored_zip,
#     is_sf_audio_data,
# )

import soundfile as sf
import cv2
import torchvision
import math

logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
            self,
            sample_rate,
            max_sample_size=None,
            min_sample_size=0,
            shuffle=True,
            pad=False,
            normalize=False,
            compute_mask_indices=False,
            **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        self.max_visual_frame = math.floor(self.max_sample_size / (self.sample_rate * 0.04))
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, audio_source, audio_target_size, visual_source, visual_target_size):
        size = visual_source.size(2)
        diff = size - visual_target_size
        if diff <= 0:
            return audio_source[:audio_target_size],visual_source.squeeze(0)[:, :]

        # # random start and end
        # v_start = np.random.randint(0, diff + 1)
        # v_end = v_start+visual_target_size
        # a_start=round((v_start/112)*0.04*self.sample_rate)
        # a_end=a_start+audio_target_size
        # return audio_source[a_start:a_end],visual_source.squeeze(0)[:,v_start:v_end]

        # start from beginning
        if not self.pad:
            return audio_source[:audio_target_size], visual_source.squeeze(0)[:, :visual_target_size]
        else:
            return audio_source[:audio_target_size], visual_source.squeeze(0)[:, :]


    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["audio_source"] is not None]
        if len(samples) == 0:
            return {}

        audio_sources = [s["audio_source"] for s in samples]
        visual_sources = [s["visual_source"] for s in samples]
        audio_sizes = [len(s) for s in audio_sources]
        visual_sizes = [s.size(-1) for s in visual_sources]

        if self.pad:
            audio_target_size = min(max(audio_sizes), self.max_sample_size)
            visual_target_size = min(max(visual_sizes), self.max_visual_frame * 112)
        else:
            # cropping
            audio_target_size = min(min(audio_sizes), self.max_sample_size)
            visual_target_size = min(min(visual_sizes), self.max_visual_frame * 112)
        audio_target_size = int((visual_target_size / 112) * 0.04 * self.sample_rate)

        collated_audio_sources = audio_sources[0].new_zeros(len(audio_sources), audio_target_size)
        collated_visual_sources = list()
        audio_padding_mask = (
            torch.BoolTensor(collated_audio_sources.shape).fill_(False) if self.pad else None
        )
        # FIXME visual在这儿不管padding，要padding的话在过完MoCo之后再padding到最长，补上padding_mask
        for i, (audio_source, audio_size, visual_source, visual_size) in enumerate(
                zip(audio_sources, audio_sizes, visual_sources, visual_sizes)):
            audio_diff = audio_size - audio_target_size
            if audio_diff == 0:
                collated_audio_sources[i] = audio_source
                collated_visual_sources.append(visual_source.squeeze(0))
            elif audio_diff < 0:
                assert self.pad
                collated_audio_sources[i] = torch.cat(
                    [audio_source, audio_source.new_full((-audio_diff,), 0.0)]
                )
                audio_padding_mask[i, audio_diff:] = True
                collated_visual_sources.append(visual_source.squeeze(0))
            else:
                collated_audio_sources[i], tmp = self.crop_to_max_size(audio_source, audio_target_size, visual_source,
                                                                           visual_target_size)
                collated_visual_sources.append(tmp.view(112, -1))


        input = {"audio_source": collated_audio_sources, "visual_source": collated_visual_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = audio_padding_mask

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class FileMMDataset(RawAudioDataset):
    def __init__(
            self,
            manifest_path,
            sample_rate,
            max_sample_size=None,
            min_sample_size=0,
            shuffle=True,
            pad=False,
            normalize=False,
            num_buckets=0,
            compute_mask_indices=False,
            **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        skipped = 0
        self.audio_fnames = []
        self.visual_fnames = []
        self.line_inds = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                sz = int(items[2])
                if (min_sample_size is not None and sz < min_sample_size) or (max_sample_size is not None and sz > max_sample_size):
                    skipped += 1
                    continue
                self.visual_fnames.append(items[0])
                self.audio_fnames.append(items[1])
                self.line_inds.add(i)
                self.sizes.append(sz)
        logger.info(
            f"loaded {len(self.visual_fnames)} visual sample, loaded {len(self.audio_fnames)} audio sample,skipped {skipped} samples")

        try:
            import pyarrow

            self.audio_fnames = pyarrow.array(self.audio_fnames)
            self.visual_fnames = pyarrow.array(self.visual_fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):

        audio_fname = os.path.join(self.root_dir, str(self.audio_fnames[index]))
        visual_fname = os.path.join(self.root_dir, str(self.visual_fnames[index]))
        wav, curr_sample_rate = sf.read(audio_fname)
        audio_feats = torch.from_numpy(wav).float()
        audio_feats = self.postprocess(audio_feats, curr_sample_rate)

        img = cv2.imread(visual_fname, cv2.IMREAD_GRAYSCALE)
        visual_feats = torchvision.transforms.functional.to_tensor(img)

        return {"id": index, "audio_source": audio_feats, "visual_source": visual_feats}


class BinarizedAudioDataset(RawAudioDataset):
    def __init__(
            self,
            data_dir,
            split,
            sample_rate,
            max_sample_size=None,
            min_sample_size=0,
            shuffle=True,
            pad=False,
            normalize=False,
            num_buckets=0,
            compute_mask_indices=False,
            **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        from fairseq.data import data_utils, Dictionary

        self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

        root_path = os.path.join(data_dir, f"{split}.root")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                self.root_dir = next(f).strip()
        else:
            self.root_dir = None

        fnames_path = os.path.join(data_dir, split)
        self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
        lengths_path = os.path.join(data_dir, f"{split}.lengths")

        with open(lengths_path, "r") as f:
            for line in f:
                sz = int(line.rstrip())
                assert (
                        sz >= min_sample_size
                ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
                self.sizes.append(sz)

        self.sizes = np.array(self.sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)
        logger.info(f"loaded {len(self.fnames)} samples")

    def __getitem__(self, index):

        fname = self.fnames_dict.string(self.fnames[index], separator="")
        if self.root_dir:
            fname = os.path.join(self.root_dir, fname)

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}
