__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import logging
from typing import Union

import numpy as np
import xarray as xr

from zarr_parallel.utils import logstream

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

class BaseTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sample):
        raise NotImplementedError("Transform must implement the __call__ method.")
    
class ReverseAxis(BaseTransform):

    def __init__(self, dim: str):
        self.dim = dim

    def __call__(self, sample: xr.Dataset) -> xr.Dataset:

        return sample.isel(**{self.dim: slice(None,None,-1)})

class TilerTransform(BaseTransform):
    """
    A transform that takes a Dataset or DataArray and breaks it into smaller tiles along specified dimensions.
    This uses the xarray `coarsen` + `construct` pattern to create non-overlapping tiles of the data, which can 
    be useful for training models on large spatial datasets by reducing memory usage and allowing for batch 
    processing of smaller chunks of data.
    """
    def __init__(
        self,
        boundary: str = "pad",
        validate_axis_order: bool = False,
        discontinuity_periods: dict[str, float] | None = None,
        **dim_tile_sizes,
    ):
        self.boundary = boundary
        self.validate_axis_order = validate_axis_order
        self.discontinuity_periods = discontinuity_periods or {"longitude": 360.0, "lon": 360.0}
        self.tile_sizes = dim_tile_sizes

    def _validate_axis_order(self, sample: xr.DataArray) -> None:
        for dim in self.tile_sizes:
            if dim not in sample.coords:
                continue
            coords = np.asarray(sample.coords[dim].values)
            if coords.ndim != 1:
                continue
            if coords.size < 2:
                continue

            diffs = np.diff(coords)
            if np.issubdtype(diffs.dtype, np.timedelta64):
                ok = np.all(diffs > np.timedelta64(0, "ns"))
            else:
                ok = np.all(diffs > 0)

            if not ok:
                raise ValueError(
                    f"Axis '{dim}' is not strictly ascending. "
                    "Either sort/reverse this axis before tiling, or set "
                    "validate_axis_order=False to bypass this guardrail."
                )

    def _validate_no_discontinuity_crossing(self, coarsened: xr.DataArray, tile_dims: dict[str, tuple[str, str]]) -> None:
        for dim, period in self.discontinuity_periods.items():
            if dim not in tile_dims:
                continue
            coarse_dim, fine_dim = tile_dims[dim]
            if dim in coarsened.coords:
                coord = coarsened[dim]
            elif fine_dim in coarsened.coords:
                coord = coarsened[fine_dim]
            else:
                continue

            if coarse_dim not in coord.dims or fine_dim not in coord.dims:
                continue

            values = np.asarray(coord.transpose(coarse_dim, fine_dim).values)
            if values.ndim != 2 or values.shape[1] < 2:
                continue
            if np.issubdtype(values.dtype, np.datetime64):
                continue

            diffs = np.abs(np.diff(values.astype(np.float64), axis=1))
            crossing = diffs > (period / 2.0)
            if np.any(crossing):
                bad_tiles = np.where(crossing.any(axis=1))[0].tolist()
                raise ValueError(
                    f"Detected tiler discontinuity crossing on axis '{dim}' "
                    f"(period={period}). Affected coarse tile ids: {bad_tiles[:10]}"
                )

    def __call__(self, sample: xr.DataArray) -> xr.DataArray:
        #check_object_type(sample, allowed_types=DA, caller=self.__class__.__name__)

        if self.validate_axis_order:
            self._validate_axis_order(sample)

        # Create the dictionary to send to the ".construct()" method, using a naming convention of
        # ("{dim}_coarse", "{dim}_fine") for the new dimensions created by the tiling process.
        tile_dims = {dim: (f"{dim}_coarse", f"{dim}_fine") for dim in self.tile_sizes}
        coarse_dims = {dim: f"{dim}_coarse" for dim in self.tile_sizes}
        fine_dims = {dim: f"{dim}_fine" for dim in self.tile_sizes}
        coarsened = sample.coarsen(**self.tile_sizes, boundary=self.boundary).construct(**tile_dims)  # type: ignore

        self._validate_no_discontinuity_crossing(coarsened, tile_dims)

        # Prepare a stacking regrouping of the original dimensions and the new dimensions
        batch_dims = []
        target_dims = []
        for dim in sample.dims:
            if dim in self.tile_sizes:
                batch_dims.append(f"{dim}_coarse")
                target_dims.append(f"{dim}_fine")
            else:
                target_dims.append(dim) 

        stacked = coarsened.stack(batch_dim=batch_dims)
        # Reorder to have batch_dim first, followed by the original dimensions and then the fine tile dimensions
        tiled = stacked.transpose("batch_dim", *target_dims)

        # Store reverse-lookup metadata in attrs
        tiled.attrs.update({
            "tiler_tile_sizes": self.tile_sizes,
            "tiler_boundary": self.boundary,
            "tiler_validate_axis_order": self.validate_axis_order,
            "tiler_discontinuity_periods": self.discontinuity_periods,
            "tiler_original_sizes": {dim: sample.sizes[dim] for dim in self.tile_sizes},
            "tiler_original_coords": {dim: sample.coords[dim].values.tolist() 
                                    for dim in self.tile_sizes if dim in sample.coords},
            "tiler_coarse_dims": coarse_dims,
            "tiler_fine_dims": fine_dims,
            "tiler_batch_dims": [coarse_dims[dim] for dim in sample.dims if dim in self.tile_sizes],
        })
        return tiled

try:
    from frame_fm.transforms import transform_mapping
    TRANSFORM_MAPPING = transform_mapping
except ImportError:
    TRANSFORM_MAPPING = {}

    # Default Transforms
    TRANSFORM_MAPPING['reverse_axis'] = ReverseAxis
    TRANSFORM_MAPPING['tiled'] = TilerTransform

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

    return offsets, ends, better_slice

def calculate_region_selection(
        ds: xr.Dataset,
        offsets: dict,
        ends: dict
    ) -> dict:
    
    dim_spec = {}

    # Determine region extents
    for dim in ds.dims:

        region_min = offsets[dim] # Offset to start of slice
        region_max = ends[dim]

        total_region = int(region_max) - int(region_min)
        dim_spec[dim] = {
            'source_min': int(region_min),
            'source_max': int(region_max),
            'total_region': total_region
        }

    return dim_spec

def apply_transforms(
        ds: xr.Dataset, 
        common_transforms: list,
        variable_transforms: dict,
        region_transform: Union[dict,None] = None

    ) -> list[xr.DataArray]:
    """
    Apply pre-transforms from the selector to the dataset

    Also calculates chunk-region offset based on existing chunks if available
    """
    offsets, ends = None,None
    dim_spec = None
    better_slice = None
    for transform in common_transforms:
        transform_type = transform.pop('type')
        if hasattr(ds, transform_type):
            if transform_type in ['sel','isel']:

                # Extract selection parameters immediately before transformations
                offsets, ends, better_slice = determine_better_selection(ds, transform)
                dim_spec = calculate_region_selection(ds, offsets, ends)

                # Transform sel to slice before applying.
                for k, v in transform.items():
                    if isinstance(v,list) or isinstance(v,tuple):
                        if len(v) == 3:
                            transform[k] = slice(v[0],v[1],v[2])
                        else:
                            transform[k] = slice(v[0],v[1])

            ds = getattr(ds, transform_type)(**transform)
        elif transform_type == 'region_isel' and region_transform is not None:
            ds = ds.isel(**region_transform)
        elif transform_type in TRANSFORM_MAPPING:
            # Apply specific custom transform here.
            ds = TRANSFORM_MAPPING[transform_type](**transform)(ds)
        else:
            raise ValueError(f'Unsupported transformation: {transform_type}')

    ds_transformed = []
    for var, vtransforms in variable_transforms.items():
        ds_var = ds[var]
        for transform in vtransforms:
            transform_type = transform.pop('type')
            ds_var = getattr(ds_var, transform_type)(**transform)
        ds_transformed.append(ds_var)

    return {'datasets':ds_transformed, 'offsets': offsets, 'ends':ends, 'dim_spec': dim_spec, 'recommendations': {'sel':better_slice}}