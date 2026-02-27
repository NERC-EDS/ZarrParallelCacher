__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import xarray as xr
from typing import Union

try:
    from frame_fm.transforms import transform_mapping
    TRANSFORM_MAPPING = transform_mapping
except ImportError:
    TRANSFORM_MAPPING = {}

def apply_transforms(
        ds: xr.Dataset, 
        common_transforms: list,
        variable_transforms: dict,
        region_transform: Union[dict,None] = None

    ) -> list[xr.DataArray]:
    """
    Apply pre-transforms from the selector to the dataset
    """

    for transform in common_transforms:
        transform_type = transform.pop('type')
        if hasattr(ds, transform_type):
            ds = getattr(ds, transform_type)(**transform)
        elif transform_type == 'region_isel' and region_transform is not None:
            ds = ds.isel(**region_transform)
        elif transform_type in TRANSFORM_MAPPING:
            # Apply specific custom transform here.
            ds = TRANSFORM_MAPPING[transform_type](ds, **transform)
        else:
            raise ValueError(f'Unsupported transformation: {transform_type}')

    ds_transformed = []
    for var, vtransforms in variable_transforms():
        ds_var = ds[var]
        for transform in vtransforms:
            transform_type = transform.pop('type')

            ds_var = getattr(ds_var, transform_type)(**transform)
        ds_transformed.append(ds_var)

    return ds_transformed