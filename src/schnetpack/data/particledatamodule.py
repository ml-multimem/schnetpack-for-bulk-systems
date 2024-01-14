from schnetpack.data.datamodule import AtomsDataModule
from typing import Optional, List, Dict, Union
import torch
from schnetpack.data import AtomsDataFormat, SplittingStrategy, BaseAtomsData, ASEParticlesData, AtomsDataError

class ParticlesDataModule(AtomsDataModule):
    """
    An extension to the AtomsDataModule for datasets containig additional
    properties with respect to ASE Atoms.

    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = False,
    ):

        super().__init__(
            datapath,
            batch_size,
            num_train,
            num_val,
            num_test,
            split_file,
            format,
            load_properties,
            val_batch_size,
            test_batch_size,
            transforms,
            train_transforms,
            val_transforms,
            test_transforms,
            num_workers,
            num_val_workers,
            num_test_workers,
            property_units,
            distance_unit,
            data_workdir,
            cleanup_workdir_stage,
            splitting,
            pin_memory,
        )

    def setup(self, stage: Optional[str] = None):
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = self.load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)
            self._setup_transforms()

    def load_dataset(self, datapath: str, format: AtomsDataFormat, **kwargs) -> BaseAtomsData:
        """
        Load dataset.

        Args:
            datapath: file path
            format: atoms data format
            **kwargs: arguments for passed to AtomsData init
        """
        if format is AtomsDataFormat.ASE:
            dataset = ASEParticlesData(datapath=datapath, **kwargs)
        else:
            raise AtomsDataError(f"Unknown format: {format}")

        return dataset