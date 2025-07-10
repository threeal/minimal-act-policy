import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


class DatasetsLoader:
    TRAIN_RATIO = 0.8

    class NormStats:
        def __init__(self, eps_files: list[Path]) -> None:
            all_controls = []
            for eps_file in eps_files:
                with h5py.File(eps_file, "r") as f:
                    names = [
                        "joint_controls_12",
                        "joint_controls_34",
                        "joint_controls_56",
                        "gripper_controls",
                    ]

                    min_length = min(f[name].shape[0] for name in names)
                    controls = np.concatenate(
                        [f[name][:min_length, 1:] for name in names], axis=1
                    )

                    all_controls.append(torch.from_numpy(controls).float())

            all_controls = torch.cat(all_controls)
            self.controls_mean = all_controls.mean(dim=[0])
            self.controls_std = torch.clip(all_controls.std(dim=[0]), 1e-2, np.inf)

        def dump(self, stats_path: str) -> None:
            with Path.open(stats_path, "wb") as f:
                stats = {
                    "controls_mean": self.controls_mean,
                    "controls_std": self.controls_std,
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
            with h5py.File(".records/record_0.h5py", "r") as f:
                cam_frames_len = min(
                    f[f"camera_frame_times_{i}"].shape[0] for i in range(2)
                )

                cam_frame_idx = np.random.default_rng().choice(cam_frames_len)
                ts = f["camera_frame_times_0"][cam_frame_idx]

                cam_frames = []
                for cam_idx in range(2):
                    data = f[f"camera_frames_{cam_idx}/{cam_frame_idx}"][:]
                    cam_frames.append(cv2.imdecode(data, cv2.IMREAD_COLOR))

                cam_frames = torch.from_numpy(np.stack(cam_frames, axis=0)).float()
                cam_frames = torch.einsum("k h w c -> k c h w", cam_frames)

                jc12 = f["joint_controls_12"][:]
                jc34 = f["joint_controls_34"][:]
                jc56 = f["joint_controls_56"][:]
                gc = f["gripper_controls"][:]

                feedback = np.zeros(7)

                jc12_i = 0
                jc34_i = 0
                jc56_i = 0
                gc_i = 0

                while jc12_i < jc12.shape[0] and jc12[jc12_i][0] <= ts:
                    feedback[0] = jc12[jc12_i][1]
                    feedback[1] = jc12[jc12_i][2]
                    jc12_i += 1

                while jc34_i < jc34.shape[0] and jc34[jc34_i][0] <= ts:
                    feedback[2] = jc34[jc34_i][1]
                    feedback[3] = jc34[jc34_i][2]
                    jc34_i += 1

                while jc56_i < jc56.shape[0] and jc56[jc56_i][0] <= ts:
                    feedback[4] = jc56[jc56_i][1]
                    feedback[5] = jc56[jc56_i][2]
                    jc56_i += 1

                while gc_i < gc.shape[0] and gc[gc_i][0] <= ts:
                    feedback[6] = gc[gc_i][1]
                    gc_i += 1

                control = feedback.copy()
                controls = np.zeros((400, 7))
                controls_i = 0

                while (
                    controls_i < controls.shape[0]
                    and jc12_i < jc12.shape[0]
                    and jc34_i < jc34.shape[0]
                    and jc56_i < jc56.shape[0]
                    and gc_i < gc.shape[0]
                ):
                    ts += 1 / 60

                    while jc12_i < jc12.shape[0] and jc12[jc12_i][0] <= ts:
                        control[0] = jc12[jc12_i][1]
                        control[1] = jc12[jc12_i][2]
                        jc12_i += 1

                    while jc34_i < jc34.shape[0] and jc34[jc34_i][0] <= ts:
                        control[2] = jc34[jc34_i][1]
                        control[3] = jc34[jc34_i][2]
                        jc34_i += 1

                    while jc56_i < jc56.shape[0] and jc56[jc56_i][0] <= ts:
                        control[4] = jc56[jc56_i][1]
                        control[5] = jc56[jc56_i][2]
                        jc56_i += 1

                    while gc_i < gc.shape[0] and gc[gc_i][0] <= ts:
                        control[6] = gc[gc_i][1]
                        gc_i += 1

                    controls[controls_i] = control
                    controls_i += 1

                cam_frames /= 255

                feedback = (
                    torch.from_numpy(feedback).float() - self.norm_stats.controls_mean
                ) / self.norm_stats.controls_std

                controls = (
                    torch.from_numpy(controls).float() - self.norm_stats.controls_mean
                ) / self.norm_stats.controls_std

                control_is_pads = np.zeros(400)
                control_is_pads[controls_i:] = 1
                control_is_pads = torch.from_numpy(control_is_pads).bool()

                return cam_frames, feedback, controls, control_is_pads

    def __init__(
        self,
        dataset_dir: Path,
        camera_names: list[str],
        batch_size: int,
    ) -> None:
        eps_files = list(dataset_dir.glob("*.h5py"))
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
