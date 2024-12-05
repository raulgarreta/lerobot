#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.default import MainConfig
from lerobot.configs.policies import PretrainedConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PretrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps config key (in-place) by using `eval`.

    Doesn't do anything if delta_timestamps is not specified or has already been resolve (as evidenced by
    the data type of its values).
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: MainConfig, split: str = "train") -> LeRobotDataset | MultiLeRobotDataset:
    """
    Args:
        cfg: A Hydra config as per the LeRobot config scheme.
        split: Select the data subset used to create an instance of LeRobotDataset.
            All datasets hosted on [lerobot](https://huggingface.co/lerobot) contain only one subset: "train".
            Thus, by default, `split="train"` selects all the available data. `split` aims to work like the
            slicer in the hugging face datasets:
            https://huggingface.co/docs/datasets/v2.19.0/loading#slice-splits
            As of now, it only supports `split="train[:n]"` to load the first n frames of the dataset or
            `split="train[n:]"` to load the last n frames. For instance `split="train[:1000]"`.
    Returns:
        The LeRobotDataset.
    """
    # A soft check to warn if the environment matches the dataset. Don't check if we are using a real world env (dora).
    if cfg.env.type != "dora":
        if isinstance(cfg.dataset.repo_id, str):
            dataset_repo_ids = [cfg.dataset.repo_id]  # single dataset
        elif isinstance(cfg.dataset.repo_id, list):
            dataset_repo_ids = cfg.dataset.repo_id  # multiple datasets
        else:
            raise ValueError(
                "Expected cfg.dataset.repo_id to be either a single string to load one dataset or a list of "
                "strings to load multiple datasets."
            )

        for dataset_repo_id in dataset_repo_ids:
            if cfg.env.type not in dataset_repo_id:
                logging.warning(
                    f"There might be a mismatch between your training dataset ({dataset_repo_id=}) and your "
                    f"environment ({cfg.env.type=})."
                )

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        # TODO (aliberts): add 'episodes' arg from config after removing hydra
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, local_files_only=cfg.dataset.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
            local_files_only=cfg.dataset.local_files_only,
        )
    else:
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        # for key, stats_dict in cfg.override_dataset_stats.items():
        #     for stats_type, listconfig in stats_dict.items():
        #         # example of stats_type: min, max, mean, std
        #         stats = OmegaConf.to_container(listconfig, resolve=True)
        #         dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
