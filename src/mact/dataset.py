import pickle
from pathlib import Path

import h5py
import numpy as np
import torch


class DatasetsLoader:
    TRAIN_RATIO = 0.8

    class NormStats:
        def __init__(self, dataset_dir: Path, num_episodes: int) -> None:
            all_qpos_data = []
            all_action_data = []
            for episode_idx in range(num_episodes):
                dataset_path = dataset_dir / f"episode_{episode_idx}.hdf5"
                with h5py.File(dataset_path, "r") as root:
                    qpos = root["/observations/qpos"][()]
                    action = root["/action"][()]
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))
            all_qpos_data = torch.stack(all_qpos_data)
            all_action_data = torch.stack(all_action_data)

            # normalize action data
            action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
            action_std = all_action_data.std(dim=[0, 1], keepdim=True)
            action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

            # normalize qpos data
            qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
            qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
            qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

            self.action_mean = action_mean.numpy().squeeze()
            self.action_std = action_std.numpy().squeeze()
            self.qpos_mean = qpos_mean.numpy().squeeze()
            self.qpos_std = qpos_std.numpy().squeeze()
            self.example_qpos = qpos

        def dump(self, stats_path: str) -> None:
            with Path.open(stats_path, "wb") as f:
                stats = {
                    "action_mean": self.action_mean,
                    "action_std": self.action_std,
                    "qpos_mean": self.qpos_mean,
                    "qpos_std": self.qpos_std,
                    "example_qpos": self.example_qpos,
                }
                pickle.dump(stats, f)

    class Dataset(torch.utils.data.Dataset):
        def __init__(
            self,
            episode_ids: list[int],
            dataset_dir: str,
            camera_names: list[str],
            norm_stats: "DatasetsLoader.NormStats",
        ) -> None:
            super().__init__()
            self.episode_ids = episode_ids
            self.dataset_dir = dataset_dir
            self.camera_names = camera_names
            self.norm_stats = norm_stats

            self.rng = np.random.default_rng()

        def __len__(self) -> int:
            return len(self.episode_ids)

        def __getitem__(self, index: int) -> any:
            sample_full_episode = False  # hardcode

            episode_id = self.episode_ids[index]
            dataset_path = self.dataset_dir / f"episode_{episode_id}.hdf5"
            with h5py.File(dataset_path, "r") as root:
                original_action_shape = root["/action"].shape
                episode_len = original_action_shape[0]
                start_ts = 0 if sample_full_episode else self.rng.choice(episode_len)
                # get observation at start_ts only
                qpos = root["/observations/qpos"][start_ts]
                image_dict = {}
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                        start_ts
                    ]

                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts

            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1

            # new axis for different cameras
            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum("k h w c -> k c h w", image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0
            action_data = (
                action_data - self.norm_stats.action_mean
            ) / self.norm_stats.action_std
            qpos_data = (
                qpos_data - self.norm_stats.qpos_mean
            ) / self.norm_stats.qpos_std

            return image_data, qpos_data, action_data, is_pad

    def __init__(
        self,
        dataset_dir: Path,
        num_episodes: int,
        camera_names: list[str],
        batch_size: int,
    ) -> None:
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(num_episodes)

        train_indices = shuffled_indices[: int(self.TRAIN_RATIO * num_episodes)]
        validate_indices = shuffled_indices[int(self.TRAIN_RATIO * num_episodes) :]

        self.norm_stats = self.NormStats(dataset_dir, num_episodes)

        self.train = torch.utils.data.DataLoader(
            self.Dataset(train_indices, dataset_dir, camera_names, self.norm_stats),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )

        self.validate = torch.utils.data.DataLoader(
            self.Dataset(validate_indices, dataset_dir, camera_names, self.norm_stats),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )
