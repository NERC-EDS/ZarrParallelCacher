__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import xarray as xr
import numpy as np
from typing import Union

try:
    from frame_fm.transforms import transform_mapping
    TRANSFORM_MAPPING = transform_mapping
except ImportError:
    TRANSFORM_MAPPING = {}

def reverse_axis(ds, dim):
    return ds.isel(**{dim:slice(None,None,-1)})

# Default Transforms
TRANSFORM_MAPPING['reverse_axis'] = reverse_axis

def determine_offsets(
        ds: xr.Dataset,
        dimensions: dict,
        reverses: dict,
    ):
    """
    Convert sel to isel for the minimum slice component.
    
    Use this to determine chunk-region offset from source.
    """
    offsets = {}
    
    for dim, chunks in ds.chunks.items():
        dim_arr = ds[dim].to_numpy()
        dinf = dimensions[dim]
        chunksize = chunks[0]

        region_min = int(np.where(dim_arr ==
            ds[dim].sel(**{dim:dinf[0], 'method':'nearest'}).to_numpy()
        )[0][0])
        offset = region_min % chunksize
        if dim in reverses:
            offset = chunksize - offset
        offsets[dim] = offset
    return offsets

def apply_transforms(
        ds: xr.Dataset, 
        common_transforms: list,
        variable_transforms: dict,
        region_transform: Union[dict,None] = None,
        reverses: dict = None

    ) -> list[xr.DataArray]:
    """
    Apply pre-transforms from the selector to the dataset

    Also calculates chunk-region offset based on existing chunks if available
    """
    reverses = reverses or {}
    offsets  = {}

    for transform in common_transforms:
        transform_type = transform.pop('type')
        if hasattr(ds, transform_type):
            if transform_type in ['sel','isel']:
                if transform_type == 'sel':
                    offsets = determine_offsets(ds, transform, reverses)
                else:
                    for dim, chunks in ds.chunks.items():
                        chunksize = chunks[0]
                        offset = transform[dim][0] % chunksize
                        if dim in reverses:
                            offset = chunksize - offset
                        offsets[dim] = offset

                for k, v in transform.items():
                    if isinstance(v,list):
                        if len(v) == 3:
                            transform[k] = slice(v[0],v[1],v[2])
                        else:
                            transform[k] = slice(v[0],v[1])


            ds = getattr(ds, transform_type)(**transform)
        elif transform_type == 'region_isel' and region_transform is not None:
            ds = ds.isel(**region_transform)
        elif transform_type in TRANSFORM_MAPPING:
            # Apply specific custom transform here.
            ds = TRANSFORM_MAPPING[transform_type](ds, **transform)
        else:
            raise ValueError(f'Unsupported transformation: {transform_type}')

    ds_transformed = []
    for var, vtransforms in variable_transforms.items():
        ds_var = ds[var]
        for transform in vtransforms:
            transform_type = transform.pop('type')
            ds_var = getattr(ds_var, transform_type)(**transform)
        ds_transformed.append(ds_var)

    if len(ds_transformed) == 1:
        return ds_transformed[0], offsets
    else:
        return ds_transformed, offsets