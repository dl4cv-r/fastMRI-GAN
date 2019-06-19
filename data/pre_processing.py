import torch
from torch import autograd

from data.transforms import to_tensor, normalize, normalize_instance


class InputTrainTransform:
    def __init__(self):
        pass

    def __call__(self, ds_slice, gt_slice, attrs, file_name, s_idx, acc_fac):
        with torch.autograd.no_grad():
            ds_slice, mean, std = normalize_instance(to_tensor(ds_slice))
            gt_slice = normalize(to_tensor(gt_slice), mean, std)

            ds_slice = ds_slice.clamp(min=-6, max=6).unsqueeze(dim=0)
            gt_slice = gt_slice.clamp(min=-6, max=6).unsqueeze(dim=0)

        return ds_slice, gt_slice, 0


class InputTestTransform:
    def __init__(self):
        pass

    def __call__(self, ds_slice, gt_slice, attrs, file_name, s_idx, acc_fac):
        assert gt_slice is None
        with torch.autograd.no_grad():
            ds_slice, mean, std = normalize_instance(to_tensor(ds_slice))
            ds_slice = ds_slice.clamp(min=-6, max=6).unsqueeze(dim=0)

        assert isinstance(file_name, str) and isinstance(s_idx, int), 'Incorrect types!'
        extra_params = dict(mean=mean, std=std, file_name=file_name, s_idx=s_idx, acc_fac=acc_fac, attrs=attrs)
        return ds_slice, extra_params  # Maybe change later.
