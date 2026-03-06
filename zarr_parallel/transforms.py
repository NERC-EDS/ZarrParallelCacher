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

def determine_better_selection(
        ds: xr.Dataset,
        dimensions: dict
    ):

    better_slice = {}
    offsets, ends = {},{}

    chunks = ds.chunks 
    for dim, dslice in dimensions.items():

        # Always calculate the offset for each dimension
        dim_arr = ds[dim].to_numpy()
        region_min = np.where(dim_arr ==
            ds[dim].sel(**{dim:dslice[0], 'method':'nearest'}).to_numpy()
        )[0][0]
        offsets[dim] = int(region_min)

        region_max = np.where(dim_arr ==
            ds[dim].sel(**{dim:dslice[1], 'method':'nearest'}).to_numpy()
        )[0][0]
        ends[dim] = int(region_max)

        # Ignore unchunked dimensions
        if len(chunks[dim]) == 1:
            continue

        # Check of the region-chunk border matches
        offset = region_min % chunks[dim][0]
        if offset != 0:
            better_slice[dim] = (dim_arr[region_min - offset], dslice[0])

    if len(better_slice.keys()) > 0:
        print('')
        print('INFO: Selection Recommendations: (for best chunking performance)')
        for dim, recommended in better_slice.items():
            print(f' > Adjust {dim} minimum from {recommended[1]} to {recommended[0]}')

        proceed = input('Proceed without recommendations? (Y/N): ')
        if proceed != 'Y':
            raise KeyboardInterrupt
    else:
        print('No chunk-based selection recommendations')

    return offsets, ends

def apply_transforms(
        ds: xr.Dataset, 
        common_transforms: list,
        variable_transforms: dict,
        region_transform: Union[dict,None] = None,
        recommend_changes: bool = False

    ) -> list[xr.DataArray]:
    """
    Apply pre-transforms from the selector to the dataset

    Also calculates chunk-region offset based on existing chunks if available
    """
    offsets, ends = None,None
    for transform in common_transforms:
        transform_type = transform.pop('type')
        if hasattr(ds, transform_type):
            if transform_type in ['sel','isel']:
                if recommend_changes:
                    # Assembler may recommend changes
                    offsets, ends = determine_better_selection(ds, transform)

                # Transform sel to slice before applying.
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

    return {'datasets':ds_transformed, 'offsets': offsets, 'ends':ends}