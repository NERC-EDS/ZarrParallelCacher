__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

# Replaces the cache_data_to_zarr function with the below code
import logging
import math
import os
from typing import Union

import dask.array as da
import dask
import numpy as np
import xarray as xr
import json

from zarr_parallel.utils import logstream, set_verbose
from zarr_parallel.dask_worker import configure_dask_deployment
from zarr_parallel.slurm import configure_slurm_deployment
from zarr_parallel.transforms import apply_transforms

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

dask.config.set({
    "distributed.scheduler.worker-ttl": "120s",
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s",
})

def divide_workers(workers, weights, dims):

    allocations = {}

    sum_weights = sum(weights)

    for x, weight in enumerate(weights):
        allocations[dims[x]] = int(workers**(weight/sum_weights))

    assert math.prod([a for a in allocations.values()]) == workers

    return allocations

class ZarrParallelAssembler:

    def __init__(
            self,
            data_uri: str,
            preprocessors: Union[list,None] = None,
            chunks: Union[dict,None] = None,
            cache_label: Union[str,None] = None,
            engine: str = 'kerchunk',
            variables: Union[list,None] = None,
            add_attrs: Union[dict,None] = None
        ):

        self.uri           = data_uri
        self.engine        = engine
        self.variables     = variables
        self.transforms    = preprocessors
        self.dimensions = None

        self.add_attrs = add_attrs

        for transform in self.transforms:
            if transform['type'] in ['sel','isel']:
                self.dimensions    = dict(transform)
                self.dimensions.pop('type')

        if self.dimensions is None:
            raise ValueError('No selection criteria applied')
        self.output_chunks = chunks

        self.cache_label = cache_label or ''

        # Derived properties
        self._ds = None

        ds = self._native_ds()
        logger.info(f'Established connection to {self.uri}')

        if variables is None:
            # All variables
            self.variables = {v : {} for v in ds.variables if v not in ds.dims}

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
            transformed = apply_transforms(
                self._native_ds(chunks=chunks),
                common_transforms=transforms,
                variable_transforms=variables,
                recommend_changes=True
            )
            self._ds = transformed['datasets']
            self.offsets = transformed['offsets']
            self.array_ends = transformed['ends']

        return self._ds
    
    def _determine_num_jobs(self, memory_limit: str) -> int:
        """
        Determine the number of jobs given the memory limit

        Divide the total array dimensions by this value to get the total 
        """

        mem_units = ['B','KB','MB','GB','TB','PB']
        bibi_units = ['z','KIB','MIB','GIB','TIB','PIB']

        suffix = memory_limit[-2:]
        if suffix[0].isnumeric():
            suffix = suffix[-1]

        mem = float(memory_limit.replace(suffix,''))
        if suffix in mem_units:
            mem*=(1000**mem_units.index(suffix.upper()))
        elif suffix in bibi_units:
            mem*=(1024**mem_units.index(suffix.upper()))
        else:
            raise ValueError(
                f'Memory Limit format unrecognised: {memory_limit} - '
                'Suffix should conform to e.g MB/GiB'
            )
        
        mem = mem/16
        
        # Increase number of minimal_arrays by 1 until memory limit is reached
        # total_array / (minimal_array*n_arrays) gives the optimal number of workers
        self._obtain_ds()
        
        min_arr, total_arr = [],[]
        for dim in self.dimensions.keys():
            minimal = 0
            total = 0
            source_chunk = self.source_chunks[dim]
            output_chunk = self.output_chunks.get(dim,source_chunk)

            min_region = math.lcm(int(output_chunk), int(source_chunk))
            position = 0
            beyond = False
            while not beyond:
                if position + source_chunk > self.offsets[dim] and position < self.array_ends[dim]:
                    total += source_chunk
                if position < min_region:
                    minimal += source_chunk
                if position - source_chunk > self.array_ends[dim]:
                    beyond = True
                position += source_chunk
            min_arr.append(minimal)
            total_arr.append(total)
            
        # Compare minimum region size with total selection size to find num jobs
        min_size = math.prod(min_arr)
        tot_size = math.prod(total_arr)

        regions_per_job = math.floor(mem/min_size)

        njobs = math.ceil(tot_size/(min_size*regions_per_job))

        logger.info(f'Dividing into {njobs} jobs for job memory limit {memory_limit}')

        return njobs

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
            dim_arr = ds[0][dim].to_numpy()

            region_min = 0 + self.offsets[dim] # Offset to start of slice
            region_max = len(dim_arr) + self.offsets[dim]

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
            rpw = max(math.ceil(total_region/(min_region*num_workers_per_dim[dim])),1)
            # RPW rounded UP so the actual workers is always LOWER than requested

            # Region_size (per worker)
            worker_size = int(rpw * min_region)

            # To go into the config file for the individual workers
            dim_spec[dim] = {
                'source_min': int(region_min),
                'source_max': int(region_max),
                'worker_size': worker_size,
                'cache_size': int(output_chunks),
                'total_region': total_region
            }

            actual_workers *= math.ceil(total_region/worker_size)

        return dim_spec, actual_workers
    
    def _determine_regional_transforms(self) -> list:
        """
        Output transforms to each region, with selection modifications.
        """
        regional_transforms = []
        for transform in self.transforms:
            if transform['type'] in ['sel','isel']:
                regional_transforms.append({'type':'region_isel'})
            else:
                regional_transforms.append(transform)
        return regional_transforms

    def _create_empty_zarr(self, worker_config: dict):
        """
        Create empty zarr based on the first of the variable dataArrays.
        """
        
        ds_transformed = self._obtain_ds()
        
        worker_dims = worker_config['region_info']

        ds_dims_only = xr.Dataset({
            d: ds_transformed[0][d] for d in worker_dims.keys()
        })
        
        # Copy global attributes
        ds_dims_only.attrs = ds_transformed[0].attrs

        if self.add_attrs is not None:
            for k,v in self.add_attrs.items():
                ds_dims_only[k] = v

        # Copy dimension encoding
        for dim in worker_dims.keys():
            encoding = ds_transformed[0][dim].encoding
            encoding.pop('chunks',None)
            encoding.pop('preferred_chunks',None)
            ds_dims_only[dim].encoding = encoding

        # Add encoding per dimension (compressors, filters etc.)

        chunks = {d: min(v['cache_size'],v['worker_size']) for d, v in worker_dims.items()}

        logger.info(f'Rechunking: {chunks}')
        ds_dims_only.chunk(chunks)

        empty_shape = [v['total_region'] for v in worker_dims.values()]
        empty_var = da.empty(empty_shape, chunks=chunks)

        encoding = {}
        for dsv in ds_transformed:
            logger.info(f'Writing empty DataArray: {dsv}')
            var = dsv.name
            ds_dims_only[var] = xr.DataArray(empty_var, dims=list(worker_dims.keys()))
            ds_dims_only[var].attrs = dsv.attrs

            # Force Zarr to use Dask chunk structure
            # Preserve non-chunk encoding attributes
            encoding[var] = {'chunks': [v['cache_size'] for d, v in worker_dims.items()]}

        ds_dims_only.to_zarr(
            worker_config['dataset']['zarr_cache'], 
            zarr_format=2, 
            compute=False, 
            consolidated=True,
            encoding=encoding
        )
        logger.info(f'Empty zarr store created at {worker_config["dataset"]["zarr_cache"]}')

    def _arrange_region_selector(
            self, 
            zarr_store: str, 
            dim_spec: dict):

        return {
            'dataset':{
                'uri': self.uri,
                'engine': self.engine,
                'kwargs':{},
                'zarr_cache': zarr_store
            },
            'common':{'pre_transforms': self._determine_regional_transforms()},
            'variables': self.variables,
            'region_info': dim_spec
        }

    def cache(
            self,
            cache_store: Union[str,object], # Can be zarr store or Pathlike?
            num_jobs: Union[int,None] = None, # Number of workers per zarr store
            generate_stats: bool = False,
            deploy_mode: str = 'SLURM',
            await_completion: bool = True,
            simultaneous_worker_limit: int = 50,
            memory_limit: str = "2GB",
            worker_timeout: str = "30:00",
            overwrite: bool = True
        ):
        """
        Method to cache selected data to a zarr store
        
        Will send out parallel workers and has option to wait for completion.
        """

        if num_jobs is None:
            num_jobs = self._determine_num_jobs(memory_limit)

        cache_dir = '/'.join(cache_store.split('/')[:-1])
        zarr_store = cache_store.split('/')[-1]

        # Handle overwriting existing store
        if os.path.isdir(cache_store):
            if overwrite:
                os.system(f'rm -rf {cache_store}')
            else:
                raise ValueError()

        if not os.path.isdir(f'{cache_dir}/temp'):
            os.makedirs(f'{cache_dir}/temp')

        dim_spec = self._determine_region_extents()

        dim_spec, actual_workers = self._determine_worker_arrangements(dim_spec, num_jobs)

        logger.info(f'Requested workers: {num_jobs}')
        logger.info(f'Actual workers: {actual_workers} (Chunk limitations)')

        worker_config = self._arrange_region_selector(cache_dir, dim_spec)

        chunks = {d: min(v['cache_size'],v['worker_size']) for d, v in worker_config['region_info'].items()}

        worker_config_file = f'{cache_dir}/temp/{zarr_store}.temp.json'

        with open(worker_config_file,'w') as f:
            json.dump(worker_config, f)

        match deploy_mode:
            case 'SLURM':

                self._create_empty_zarr(worker_config)

                status = configure_slurm_deployment(
                    cache_dir,
                    zarr_store,
                    worker_config_file,
                    actual_workers,
                    simultaneous_worker_limit=simultaneous_worker_limit,
                    worker_timeout=worker_timeout,
                    memory_limit=memory_limit,
                    await_completion=await_completion,
                    chunks=chunks
                )
            case 'dask_distributed':

                self._create_empty_zarr(worker_config)
                
                # Cluster workers set via limit
                # Number of jobs is now number of traditional workers
                status = configure_dask_deployment(
                    num_dask_workers=simultaneous_worker_limit,
                    job_ids=actual_workers,
                    worker_config_file=worker_config_file,
                    memory_limit=memory_limit,
                    threads_per_worker=1
                )

            case 'series':
                # Serial cacher for very small datasets.
                ds_transformed = self._obtain_ds()
                for ds in ds_transformed:

                    ds.chunk(chunks)

                    ds.to_zarr(
                        zarr_store, 
                        compute=True,
                        zarr_format=2, 
                        consolidated=True,
                        write_empty_chunks=True,
                        mode='w')
                
                status = True

        if not status:
            raise ValueError
