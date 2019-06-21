from pathlib import Path
import random

import h5py
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, root, transform, acc_fac=None, training=True):
        super().__init__()

        self.root = root
        self.transform = transform
        self.acc_fac = acc_fac
        self.training = training

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        file_names = list(Path(root).iterdir())
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise FileNotFoundError("Sorry! No files present in this directory.")

        print(f'Initializing {Path(root).name}.')
        slice_counts = [self.get_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()

        if self.acc_fac is not None:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, self.acc_fac] for s_idx in range(slice_num)]

        else:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, random.choice((4, 8))] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):
        with h5py.File(name=file_name, mode='r') as hf:
            try:  # Train and Val
                return hf['1'].shape[0]
            except KeyError:  # Test
                return hf['data'].shape[0]

    def h5_slice_parse_fn(self, file_name, s_idx, acc_fac):
        with h5py.File(file_name, mode='r') as hf:
            try:  # Train & Val
                ds_slice_arr = hf[str(acc_fac)][s_idx]
            except KeyError:  # Test
                ds_slice_arr = hf['data'][s_idx]
            try:  # Train & Val
                gt_slice_arr = hf['1'][s_idx] if self.training else None  # Reduce data I/O
            except KeyError:  # Test
                gt_slice_arr = None
            attrs = dict(hf.attrs)
        return ds_slice_arr, gt_slice_arr, attrs

    def __getitem__(self, idx):  # Need to add transforms.
        file_name, s_idx, acc_fac = self.names_and_slices[idx]
        ds_slice, gt_slice, attrs = self.h5_slice_parse_fn(file_name, s_idx, acc_fac)
        return self.transform(ds_slice, gt_slice, attrs, Path(file_name).name, s_idx, acc_fac)
