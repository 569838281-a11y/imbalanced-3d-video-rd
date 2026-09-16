from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

#from torchvision.transforms import transforms
#from erdes.data.components.utils import resize
from erdes.data.collate import pad_video_batch_collate
from erdes.data.components.cached_video_dataset import CachedVideoDataset
from erdes.data.components.erdes_dataset import VideoDataset


class ERDESDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        size: Tuple[int, int, int],
        data_root: str = "",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_dir: str = "",
        use_cache: bool = False,
        persistent_workers: bool = False,
        defer_resize: bool = False,
        cache_mmap: bool = True,
        use_um: bool = False,
        um_strength: float = 1.5,
        um_cache_dir: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.size = size
        self.data_root = data_root
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        #self.transforms = resize(size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass  # No downloading necessary

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train:
            self.data_train = self._make_dataset(self.train_csv, apply_um=bool(self.hparams.use_um))

        if not self.data_val:
            self.data_val = self._make_dataset(self.val_csv, apply_um=bool(self.hparams.use_um))

        if not self.data_test:
            self.data_test = self._make_dataset(self.test_csv, apply_um=False)

    def _make_dataset(self, csv_path: str, apply_um: bool = False) -> Dataset:
        um_cache = (self.hparams.um_cache_dir or "").strip()
        precomputed_um = bool(apply_um and um_cache and self.hparams.use_cache)
        um_kw = {
            "use_um": False if precomputed_um else apply_um,
            "um_strength": float(self.hparams.um_strength),
        }
        cache_dir = um_cache if precomputed_um else self.hparams.cache_dir
        if self.hparams.use_cache and cache_dir:
            return CachedVideoDataset(
                csv_path=csv_path,
                size=self.size,
                data_root=self.data_root,
                cache_dir=cache_dir,
                strict_cache=True,
                cache_mmap=bool(self.hparams.cache_mmap),
                **um_kw,
            )
        return VideoDataset(
            csv_path=csv_path,
            size=self.size,
            data_root=self.data_root,
            defer_resize=bool(self.hparams.defer_resize),
            **um_kw,
        )

    def _loader_kwargs(self) -> dict:
        nw = self.hparams.num_workers
        kwargs = {
            "num_workers": nw,
            "pin_memory": self.hparams.pin_memory,
        }
        if nw > 0:
            kwargs["persistent_workers"] = self.hparams.persistent_workers
            kwargs["prefetch_factor"] = 2
        if self.hparams.defer_resize and not self.hparams.use_cache:
            kwargs["collate_fn"] = pad_video_batch_collate
        return kwargs

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            **self._loader_kwargs(),
        )

if __name__ == "__main__":
    datamodule = ERDESDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        test_csv="test.csv",
        size=(128, 128, 128),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
)