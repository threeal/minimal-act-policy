import pickle
from pathlib import Path

import h5py
import numpy as np
import torch


class DatasetsLoader:
    TRAIN_RATIO = 0.8

    class NormStats:
        def __init__(self, eps_files: list[Path]) -> None:
            all_qpos_data = []
            all_action_data = []
            for eps_file in eps_files:
                with h5py.File(eps_file, "r") as root:
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
            eps_files: list[Path],
            camera_names: list[str],
            norm_stats: "DatasetsLoader.NormStats",
        ) -> None:
            super().__init__()
            self.eps_files = eps_files
            self.camera_names = camera_names
            self.norm_stats = norm_stats
            self.rng = np.random.default_rng()

        def __len__(self) -> int:
            return len(self.eps_files)

        def __getitem__(self, index: int) -> any:
            sample_full_episode = False  # hardcode

            with h5py.File(self.eps_files[index], "r") as root:
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
        camera_names: list[str],
        batch_size: int,
    ) -> None:
        eps_files = list(dataset_dir.glob("*.hdf5"))
        eps_files = np.random.default_rng().permutation(eps_files)

        train_eps_files = eps_files[: int(self.TRAIN_RATIO * len(eps_files))]
        validate_eps_files = eps_files[int(self.TRAIN_RATIO * len(eps_files)) :]

        self.norm_stats = self.NormStats(eps_files)

        self.train = torch.utils.data.DataLoader(
            self.Dataset(train_eps_files, camera_names, self.norm_stats),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )

        self.validate = torch.utils.data.DataLoader(
            self.Dataset(validate_eps_files, camera_names, self.norm_stats),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )
