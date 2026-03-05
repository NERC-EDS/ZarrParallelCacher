__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

# Replaces the cache_data_to_zarr function with the below code
import logging
import math
import os
from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
import json

from zarr_parallel.utils import logstream
from zarr_parallel.dask_worker import configure_dask_deployment
from zarr_parallel.slurm import configure_slurm_deployment
from zarr_parallel.transforms import apply_transforms

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

def divide_workers(workers, weights, dims):

    allocations = {}

    sum_weights = sum(weights)

    for x, weight in enumerate(weights):
        allocations[dims[x]] = int(workers**(weight/sum_weights))

    assert math.prod([a for a in allocations.values()]) == workers

    return allocations

class ZarrParallelAssembler:

    def __init__(self, selector: dict, cache_label: Union[str,None] = None):

        # Subject to change - not based on any established scheme for selectors
        self.uri           = selector['uri']
        self.engine        = 'kerchunk' # Discuss an option for this?
        self.variables     = selector['variables']
        self.transforms    = selector['common']['pre_transforms']
        self.dimensions = None
        for transform in self.transforms:
            if transform['type'] in ['sel','isel']:
                self.dimensions    = dict(transform)
                self.dimensions.pop('type')

        #self.offsets, self.reverses = self._determine_worker_offsets()

        if self.dimensions is None:
            raise ValueError('No selection criteria applied')
        self.output_chunks = selector['common'].get('chunks')

        self.cache_label = cache_label or ''

        # Derived properties
        self._ds = None
        self.offsets = None

        ds = self._native_ds()
        logger.info(f'Established connection to {self.uri}')

        # Derive source chunks and determine which are sub-chunked
        self.source_chunks = {}
        self.chunked_dims = {}
        for var in self.variables.keys():
            # Currently ignores different chunk structures between variables
            # Assumes chunk structures are the same for all variables in this selector.

            self.source_chunks = {}
            self.chunked_dims[var] = []
            for dx, chunkset in enumerate(ds[var].chunks):
                dim = ds[var].dims[dx]
                self.source_chunks[dim] = chunkset[0]
                if len(chunkset) > 1:
                    self.chunked_dims[var].append(dim)

        chunked_dims = set([tuple(c) for c in self.chunked_dims.values()])
        if len(chunked_dims) > 1:
            raise ValueError('Parallel caching not supported for different structures within the same zarr store')
        self.chunked_dims = list(chunked_dims)[0]

        # If the source chunks differ between variables, raise an 
        # error as this is not yet supported.

    def zarr_store(self) -> str:
        return f'{self.uri.split("/")[-1].split(".")[0]}_{"_".join(self.variables.keys())}{self.cache_label}.zarr'
    
    def zarr_store_path(self, cache_dir: str) -> str:
        return f'{cache_dir}/{self.zarr_store()}'
    
    def _native_ds(self, chunks: Union[str,dict,None] = 'auto') -> xr.Dataset:
        return xr.open_dataset(self.uri, engine=self.engine, chunks=chunks)

    def _obtain_ds(self, chunks: Union[str,dict,None] = 'auto') -> xr.Dataset:
        """
        Obtain the xarray dataset source object for the parent dataset
        """
        import copy
        transforms = copy.deepcopy(self.transforms)
        variables = copy.deepcopy(self.variables)

        if self._ds is None:
            self._ds, self.offsets = apply_transforms(
                self._native_ds(chunks=chunks),
                common_transforms=transforms,
                variable_transforms=variables,
                reverses=self._determine_worker_reverses()
            )

        return self._ds

    def _determine_worker_reverses(self):
        """
        Determine which dimensions have reversed worker offsets
        """
        return [transform['dim'] for transform in self.transforms if transform['type'] == 'reverse_axis']

    def _determine_region_extents(self) -> dict:
        """
        Determine from the provided selector the specs of the dim.
        
        This includes the region minimum and maximum extents in terms
        of the size of the array. This converts all coordinate selections
        to indexed values from 0 to the length of the array."""

        ds = self._obtain_ds()
        dim_spec = {}

        # Determine region extents
        for dim, dinf in self.dimensions.items():
            dim_arr = ds[dim].to_numpy()

            region_min = 0
            region_max = len(dim_arr)

            total_region = int(region_max) - int(region_min)
            dim_spec[dim] = {
                'source_min': int(region_min),
                'source_max': int(region_max),
                'total_region': total_region
            }

        return dim_spec
    
    def _determine_worker_arrangements(self, dim_spec: dict, num_workers: int) -> tuple:
        """
        Determine best arrangement for worker region extents based on source and 
        destination chunk arrangements.
        """

        chunk_tots     = sum([dim_spec[c]['total_region'] for c in self.chunked_dims])

        dim_weights    = [(dim_spec[c]['total_region']/chunk_tots) if c in self.chunked_dims else 0 for c in dim_spec.keys()]

        num_workers_per_dim = divide_workers(
            num_workers, 
            dim_weights,
            list(dim_spec.keys())
        )
        actual_workers = 1
        for dim, dinf in self.dimensions.items():

            total_region = dim_spec[dim]['total_region']
            region_min   = dim_spec[dim]['source_min']
            region_max   = dim_spec[dim]['source_max']

            # Underlying chunk structure is now always known, as it can be derived from xarray.

            # Should be doing this per var?
            source_chunks = self.source_chunks[dim]
            output_chunks = self.output_chunks.get(dim, source_chunks)

            min_region = math.lcm(int(output_chunks), int(source_chunks))

            # Unresolvable chunk structure means we cannot subdivide between workers for this dimension
            if min_region > total_region:
                min_region = total_region

            # Regions per worker - will resolve to 1 if min_region == total_region
            rpw = max(math.floor(total_region/(min_region*num_workers_per_dim[dim])),1)

            # Region_size (per worker)
            worker_size = int(rpw * min_region)

            chunk_sum = 0
            while chunk_sum <= region_min:
                chunk_sum += int(source_chunks)
            chunk_sum -= int(source_chunks)

            # Difference between chunk border and region selected
            worker_offset = self.offsets[dim]

            # To go into the config file for the individual workers
            dim_spec[dim] = {
                'source_min': int(region_min),
                'source_max': int(region_max),
                'worker_size': worker_size,
                'worker_offset': worker_offset, # Negative value
                'cache_size': int(output_chunks),
                'total_region': total_region
            }

            actual_workers *= int(total_region/worker_size)

        return dim_spec, actual_workers
    
    def _create_empty_zarr(self, worker_config: dict, vars: list, return_ds: bool = False):

        ds_transformed = self._obtain_ds()
        
        worker_dims = worker_config['region_info']

        ds_dims_only = xr.Dataset({
            d: ds_transformed[d].isel(**{
                d: slice(v['source_min'],v['source_max']) 
            }) for d, v in worker_dims.items()
        })
        
        # Copy global attributes
        ds_dims_only.attrs = ds_transformed[0].attrs

        # Copy dimension encoding
        for dim in worker_dims.keys():
            encoding = ds_transformed[0][dim].encoding
            encoding.pop('chunks',None)
            encoding.pop('preferred_chunks',None)
            ds_dims_only[dim].encoding = encoding

        # Add encoding per dimension (compressors, filters etc.)

        chunks = {d: int(v['total_region']/v['cache_size']) for d, v in worker_dims.items()}

        logger.info(f'Rechunking: {chunks}')
        ds_dims_only.chunk(chunks)

        empty_shape = [v['total_region'] for v in worker_dims.values()]
        empty_var = da.empty(empty_shape, chunks=chunks)

        var = ds_transformed.name
        ds_dims_only[var] = xr.DataArray(empty_var, dims=list(worker_dims.keys()))
        ds_dims_only[var].attrs = ds_transformed.attrs

        # Force Zarr to use Dask chunk structure
        # Preserve non-chunk encoding attributes
        encoding = {
            var: {'chunks': [v['cache_size'] for d, v in worker_dims.items()]}
        }

        ds_dims_only.to_zarr(
            worker_config['dataset']['zarr_cache'], 
            zarr_format=2, 
            compute=False, 
            consolidated=True,
            encoding=encoding
        )
        logger.info(f'Empty zarr store created at {worker_config["dataset"]["zarr_cache"]}')

        if return_ds:
            return ds_transformed

    def _arrange_region_selector(
            self, 
            cache_dir: str, 
            dim_spec: dict):

        return {
            'dataset':{
                'uri': self.uri,
                'engine': self.engine,
                'kwargs':{},
                'zarr_cache': self.zarr_store_path(cache_dir)
            },
            'common':{'pre_transforms': self.transforms + [{'type':'region_isel'}]},
            'variables': self.variables,
            'region_info': dim_spec
        }

    def cache(
            self,
            num_workers: int, # Number of workers per zarr store
            cache_dir: Union[str,object], # Can be zarr store or Pathlike?
            generate_stats: bool = False,
            deploy_mode: str = 'SLURM',
            await_completion: bool = True,
            simultaneous_worker_limit: int = 50,
            memory_limit: str = "2GB",
            worker_timeout: str = "30:00",
        ):
        """
        Method to cache selected data to a zarr store
        
        Will send out parallel workers and has option to wait for completion.
        """

        if os.path.isdir(self.zarr_store_path(cache_dir)):
            os.system(f'rm -rf {self.zarr_store_path(cache_dir)}')
        if not os.path.isdir(f'{cache_dir}/temp'):
            os.makedirs(f'{cache_dir}/temp')

        dim_spec = self._determine_region_extents()

        dim_spec, actual_workers = self._determine_worker_arrangements(dim_spec, num_workers)

        logger.info(f'Requested workers: {num_workers}')
        logger.info(f'Actual workers: {actual_workers} (Chunk limitations)')

        worker_config = self._arrange_region_selector(cache_dir, dim_spec)

        chunks = {d: min(v['cache_size'],v['worker_size']) for d, v in worker_config['region_info'].items()}

        worker_config_file = self.zarr_store_path(cache_dir).replace('zarr_cache','zarr_cache/temp') + '.json'

        with open(worker_config_file,'w') as f:
            json.dump(worker_config, f)

        match deploy_mode:
            case 'SLURM':

                self._create_empty_zarr(worker_config, [v for v in self.variables.keys()])

                status = configure_slurm_deployment(
                    cache_dir,
                    self.zarr_store(),
                    worker_config_file,
                    actual_workers,
                    simultaneous_worker_limit=simultaneous_worker_limit,
                    worker_timeout=worker_timeout,
                    memory_limit=memory_limit,
                    await_completion=await_completion,
                    chunks=chunks
                )
            case 'dask_distributed':

                # Attempting to use in-built dask-cluster parallelisation
                status = configure_dask_deployment(
                    ds=self._obtain_ds(chunks={}),
                    zarr_path=self.zarr_store_path(cache_dir),
                    num_dask_workers=simultaneous_worker_limit,
                    #job_ids=num_workers,
                    #worker_config=worker_config,
                    memory_limit=memory_limit,
                    chunks=chunks
                )

            case 'series':
                # Serial cacher for very small datasets.
                ds_transformed = self._obtain_ds()

                ds_transformed.chunk(chunks)

                ds_transformed.to_zarr(
                    self.zarr_store_path(cache_dir), 
                    compute=True,
                    zarr_format=2, 
                    consolidated=True,
                    write_empty_chunks=True,
                    mode='w')
                
                status = True

        if not status:
            raise ValueError

if __name__ == '__main__':
    selectors= [
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
                "common": {
                    "pre_transforms": [
                        {"type": "reverse_axis", "dim": "latitude"},
                        #{"type": "roll", "dim": "longitude", "shifts": None}, # Roll required BEFORE subsetting
                        {"type": "sel", "time": ["2000-03-02 00:00:00", "2010-01-10 23:00:00"], "latitude": [60, 67.8], "longitude": [10, 137.8]},
                        # xarray-based transformations SHOULDN'T affect the region arrangements.
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48},  # Example chunking strategy, can be adjusted as needed,
                },
                "variables": {
                    "d2m": [
                        {"type": "rename", "new_name_or_name_dict":"dewpoint_temperature"},
                    ],
                }
            },
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2t_repack.kr1.0.json",
                "common": {
                    "subset": {
                        "time": ["2005-01-01 11:00:00", "2008-01-10 23:00:00"],
                        "latitude": [60, -67.8],
                        "longitude": [10, 137.8]
                    },
                    "pre_transforms": [
                        {"type": "rename", "var_id": "t2m", "new_name": "surface_temperature"},
                        {"type": "roll", "dim": "longitude", "shift": None},
                        {"type": "reverse_axis", "dim": "latitude"}
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48}  # Example chunking strategy, can be adjusted as needed
                },
                "variables": {
                    "t2m": {},
                }
            }]
    
    zp = ZarrParallelAssembler(selector=selectors[0], cache_label='v2')
    zp.cache(num_workers=50,cache_dir='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache',deploy_mode='dask_distributed',simultaneous_worker_limit=4)