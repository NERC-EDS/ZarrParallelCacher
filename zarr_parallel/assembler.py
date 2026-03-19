__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import copy
import json
import logging
import math
import os
from typing import Union

import dask
import dask.array as da
import numpy as np
import xarray as xr

from zarr_parallel.dask_worker import configure_dask_deployment
from zarr_parallel.slurm import configure_slurm_deployment
from zarr_parallel.transforms import apply_transforms
from zarr_parallel.utils import logstream, set_verbose

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

dask.config.set({
    "distributed.scheduler.worker-ttl": "120s",
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "60s",
})

def divide_workers(workers: int, weights: list, dims: list) -> dict:
    """
    Split workers between dimensions based on weights.
    
    Product of worker splits must equal total workers."""

    allocations = {}

    sum_weights = sum(weights)

    for x, weight in enumerate(weights):
        allocations[dims[x]] = int(workers**(weight/sum_weights))

    assert math.prod([a for a in allocations.values()]) == workers

    return allocations

class ZarrParallelAssembler:
    description = "Class to handle parallel assembly of zarr datasets based on source chunk structure and selection/transforms."

    def __init__(
            self,
            data_uri: str,
            preprocessors: Union[list,None] = None,
            chunks: Union[dict,str,None] = None,
            cache_label: Union[str,None] = None,
            variables: Union[list,None] = None,
            add_attrs: Union[dict,None] = None,
            engine: str = 'kerchunk',
            log_level: int = 0,
        ):

        set_verbose(log_level)

        self.uri           = data_uri
        self.engine        = engine
        self.variables     = variables
        self.recommendations = {}
        self.cache_label = cache_label or ''

        self.add_attrs = add_attrs

        # Future properties
        self.reconfigure: str      = None
        self.tiler_transform: dict = None
        self._ds: xr.Dataset       = None
        self.dimensions: dict      = None
        self.output_chunks: dict   = None
        self.batch_dim_worker_size: int = None
        self.global_attrs: dict    = None
        self.dim_spec: dict        = None

        self.source_chunks: dict = None
        self.chunked_dims: dict = None
        
        # Fill in all parameters based on preprocessors and chunks
        self._interpret_params(transforms=preprocessors, chunks=chunks)

    def _interpret_params(
            self, 
            transforms: Union[list,None] = None, 
            chunks: Union[dict,str,None] = None
           ) -> list:
        """
        Interpret the transforms to determine the dimensions, 
        source chunks and other information required for tiling and selection arrangements.
        """

        ### 0. Establish connection to source endpoint

        ds = self._native_ds()
        self.global_attrs = ds.attrs
        logger.info(f'Established connection to {self.uri}')

        ### 1. Interpret transforms
        self.transforms = transforms or []
        var_mapping = {}

        subset = False
        for transform in self.transforms:
            if transform['type'] == 'subset':
                transform['type'] = 'sel'
            if transform['type'] in ['sel','isel']:
                self.dimensions    = dict(transform)
                self.dimensions.pop('type')
                subset = True

            if transform['type'] == 'tiled':
                self.reconfigure = 'tiled'
                self.tiler_transform = copy.deepcopy(transform)
                self.tiler_transform.pop('type')

            if transform['type'] == 'rename':
                var_mapping[transform['var_id']] = transform['new_name']
                
        ### 1.1 Default subset - global dataset
        if not subset:
            self.dimensions = {d: [0, len(ds[d])] for d in ds.dims}
            self.transforms.append({'type':'isel', **self.dimensions})

        ### 2. Collect all variables if not specified, including renamed ones.
        if self.variables is None:
            self.variables = {}
            for v in ds.variables:
                if v in ds.dims:
                    continue
                if v in var_mapping:
                    self.variables[var_mapping[v]] = {}
                else:
                    self.variables[v] = {}

        ### 3. Derive source chunks and determine which are sub-chunked
        chunked_dims = {}
        for var in self.variables.keys():
            source_chunks = {}
            chunked_dims[var] = []
            for dx, chunkset in enumerate(ds[var].chunks):
                dim = ds[var].dims[dx]
                source_chunks[dim] = chunkset[0]
                if len(chunkset) > 1:
                    chunked_dims[var].append(dim)

            if self.source_chunks is None:
                self.source_chunks = source_chunks
            
            if self.source_chunks != source_chunks:
                raise ValueError('Parallel caching not supported for different structures within the same zarr store')

        chunked_dims = set([tuple(c) for c in chunked_dims.values()])
        if len(chunked_dims) > 1:
            raise ValueError('Parallel caching not supported for different structures within the same zarr store')
        self.chunked_dims = list(chunked_dims)[0]

        ### 4. Interpret final chunks if needed.

        if isinstance(chunks, str) and chunks != 'auto':
            raise ValueError('Unsupported chunking scheme. Provide a dict of "{dim:chunk_size}" or use "auto" to keep source chunking')
        
        self.output_chunks = chunks

    def _recommend_tiling(self, ds: xr.Dataset):
        """
        Recommend tiling if the source chunk structure is either incompatible with the tiling scheme or there
        are simple improvements.
        """

        tiling_recommends = {'order':None, 'size':{}}

        reorder = False
        nchunks = {d: len(chunks) for d, chunks in ds.chunks.items()}
        nchunks = sorted(nchunks.items(), key=lambda x: x[1], reverse=True)
        for dx, (dim, _) in enumerate(nchunks):
            if list(self.tiler_transform.keys())[dx] != dim:
                reorder = True

        if reorder:
            tiling_recommends['order'] = (list(set(self.tiler_transform.keys())), list([d for d, _ in nchunks]))

        for dx, (dim, chunk) in enumerate(ds.chunks.items()):
            tile = self.tiler_transform.get(dim,None)

            if tile is None:
                continue

            rem = tile%chunk[0]
            if rem == 0 or rem == tile: 
                continue

            # Tiling recommendations should bring tile size in line with chunks
            if rem < chunk[0]/2:
                tiling_recommends['size'][dim] = (tile, tile - rem)
            else:
                tiling_recommends['size'][dim] = (tile, tile + (chunk[0] - rem))

        self.recommendations['tiling'] = tiling_recommends
            
    def _display_recommendations(self):
        """
        Display recommendations for improving tiling and selection arrangements.

        This will print messages, so users will always see recommendations,
        regardless of their log level.
        """
        recommended = False
        if len(self.recommendations.get('sel',{}).keys()) > 0:
            recommended = True
            print('Selection recommendations:')
            for dim, recommended in self.recommendations['sel'].items():
                print(f' > Adjust {dim} minimum from {recommended[1]} to {recommended[0]}')
        
        order = self.recommendations.get('tiling',{}).get('order')
        sizes = self.recommendations.get('tiling',{}).get('size',{})
        if order is not None or len(sizes.keys()) > 0:
            recommended = True
            print('Tiling recommendations:')
            if order is not None:
                print(f' > Adjust order of tiles from {order[0]} to {order[1]}')
            if len(sizes.keys()) > 0:
                for dim, size in sizes.items():
                    print(f' > Adjust tile size for {dim} from {size[0]} to {size[1]}')

        if recommended:
            ask = input('Continue without recommendations? (y/n) ')
            if ask.lower() != 'y':
                raise ValueError('Exiting to allow adjustments based on recommendations')
        else:
            logger.info('No recommendations for improving tiling or selection arrangements')

    def _native_ds(self, chunks: Union[str,dict,None] = 'auto') -> xr.Dataset:
        """
        Open native dataset with no transforms.
        """
        return xr.open_dataset(self.uri, engine=self.engine, chunks=chunks)

    def _transform_ds(self, chunks: Union[str,dict,None] = 'auto'):
        """
        Obtain the xarray dataset source object for the parent dataset
        """

        transforms = copy.deepcopy(self.transforms)
        variables = copy.deepcopy(self.variables)

        if self._ds is None:
            transformed = apply_transforms(
                self._native_ds(chunks=chunks),
                common_transforms=transforms,
                variable_transforms=variables
            )
            self._ds        = transformed['datasets']
            self.offsets    = transformed['offsets']
            self.array_ends = transformed['ends']
            self.dim_spec   = transformed['dim_spec']
            self.recommendations['sel'] = transformed['recommendations']['sel']
    
    # def _determine_num_jobs(self, memory_limit: str) -> int:
    #     """
    #     Determine the number of jobs given the memory limit

    #     Unused function, as it has been determined that jobs should
    #     be split at the worker-level and not at this higher level. This
    #     reduces overheads with setting up small parallel-writes as separate jobs.

    #     """
        
    #     mem = interpret_mem_limit(memory_limit)/16
        
    #     # Increase number of minimal_arrays by 1 until memory limit is reached
    #     # total_array / (minimal_array*n_arrays) gives the optimal number of workers
        
    #     min_arr, total_arr = [],[]
    #     for dim in self.dimensions.keys():
    #         minimal = 0
    #         total = 0
    #         source_chunk = self.source_chunks[dim]
    #         output_chunk = self.output_chunks.get(dim,source_chunk)

    #         min_region = math.lcm(int(output_chunk), int(source_chunk))
    #         position = 0
    #         beyond = False
    #         while not beyond:
    #             if position + source_chunk > self.offsets[dim] and position < self.array_ends[dim]:
    #                 total += source_chunk
    #             if position < min_region:
    #                 minimal += source_chunk
    #             if position - source_chunk > self.array_ends[dim]:
    #                 beyond = True
    #             position += source_chunk
    #         min_arr.append(minimal)
    #         total_arr.append(total)
            
    #     # Compare minimum region size with total selection size to find num jobs
    #     min_size = math.prod(min_arr)
    #     tot_size = math.prod(total_arr)

    #     regions_per_job = math.floor(mem/min_size)

    #     njobs = math.ceil(tot_size/(min_size*regions_per_job))

    #     logger.info(f'Dividing into {njobs} jobs for job memory limit {memory_limit}')

    #     # New setup - parallelise 

    #     return njobs
    
    def _determine_worker_arrangements(self, num_workers: int) -> tuple:
        """
        Determine best arrangement for worker region extents based on source and 
        destination chunk arrangements.
        """

        chunk_tots     = sum([self.dim_spec[c]['total_region'] for c in self.chunked_dims])

        dim_weights    = [(self.dim_spec[c]['total_region']/chunk_tots) if c in self.chunked_dims else 0 for c in self.dim_spec.keys()]

        num_workers_per_dim = divide_workers(
            num_workers, 
            dim_weights,
            list(self.dim_spec.keys())
        )
        actual_workers = 1
        for dim in self.dimensions.keys():

            total_region = self.dim_spec[dim]['total_region']
            region_min   = self.dim_spec[dim]['source_min']
            region_max   = self.dim_spec[dim]['source_max']

            # Underlying chunk structure is now always known, as it can be derived from xarray.

            # Should be doing this per var?
            source_chunks = self.source_chunks[dim]
            output_chunks = self._output_chunks().get(dim, source_chunks)

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
            self.dim_spec[dim] = {
                'source_min': int(region_min),
                'source_max': int(region_max),
                'worker_size': worker_size,
                'cache_size': int(output_chunks),
                'total_region': total_region
            }

            actual_workers *= math.ceil(total_region/worker_size)

        return actual_workers
    
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

    def _reconfigure_regions(self, num_workers: int) -> tuple:
        """
        Reconfigure regions based on the reconfigure parameter
        """

        actual_workers = num_workers

        region_info = {}
        match self.reconfigure:
            case 'tiled':

                ds = self._ds[0]
                # Total region is the full size of the array at this point in each fine_dim

                primary_dim = list(self.tiler_transform.keys())[0]

                # Primary dimension for batch dim parallelisation
                self.batch_dim_worker_size = math.ceil(
                    self.dim_spec[primary_dim]['total_region']/self.tiler_transform[primary_dim]
                )

                # Limit batch dim worker size given the number of workers
                if self.batch_dim_worker_size > len(ds.batch_dim)/num_workers:
                    self.batch_dim_worker_size = int(len(ds.batch_dim)/num_workers)
                
                # Limit number of workers given limiting batch dim worker size
                if len(ds.batch_dim)/self.batch_dim_worker_size < num_workers:
                    actual_workers =  int(len(ds.batch_dim)/self.batch_dim_worker_size)

                batch_dim = {
                    'batch_dim':{
                        'total_region':len(ds.batch_dim), 
                        'source_min':0,
                        'source_max':len(ds.batch_dim),
                        'worker_size':self.batch_dim_worker_size
                    }
                }

                fine_dim_set = [f'{d}_fine' for d in self.tiler_transform.keys()]

                fine_dims = {dim: {
                    'total_region': len(self._ds[0][dim]),
                    'source_min': 0,
                    'source_max': len(self._ds[0][dim]),
                    'worker_size': len(self._ds[0][dim])
                } for dim in fine_dim_set}

                # Original dimensions
                dimensional_vars = [d for d in self.dimensions.keys()]
                
                region_info = {
                    'dims': batch_dim,
                    'fine_dims': fine_dims,
                    'coords': dimensional_vars
                }

        return region_info, actual_workers

    def _output_chunks(self) -> dict:
        """
        Assemble output chunks
        
        If no output chunking is defined, leave empty.
        If one or more dimensions are defined, fill chunks for all dimensions.
        Chunk size defined either by user, or as the minimum of source chunk size and new region size.
        """

        if self.reconfigure == 'tiled':
            return { 'batch_dim':1 }

        if self.output_chunks == {}:
            return {}
        
        output_chunks = {}
        for dim in self.dimensions.keys():

            dim_limit=1e9
            if self.dim_spec is not None:
                dim_limit = self.dim_spec[dim]['total_region']

            output_chunk = None
            if isinstance(self.output_chunks, dict):
                output_chunk = self.output_chunks.get(dim,None)

            output_chunks[dim] = output_chunk or min(self.source_chunks[dim], dim_limit)

        return output_chunks
 
    def _create_empty_zarr(self, worker_config: dict):
        """
        Create empty zarr based on the first of the variable dataArrays.

        :param worker_config: dict. 
        """
        
        ds_transformed = self._ds

        worker_dims = worker_config['region_info']['dims']
        # worker_scalars = worker_config['region_info']['scalars']
        default_coords = worker_config['region_info'].get('coords',[])

        # Dimensions of the tiled or un-tiled dataset (configured above)
        data_vars = {d: ds_transformed[0][d] for d in worker_dims.keys()}
        all_dims = copy.deepcopy(data_vars)
        if self.reconfigure:
            all_dims.update({d: ds_transformed[0][d] for d in worker_config['region_info']['fine_dims'].keys()})

        ds_dims_only = xr.Dataset(data_vars=data_vars)
        
        ds_dims_only = self._override_global_attrs(ds_dims_only)

        # Copy dimension encoding
        for dim in data_vars.keys():
            encoding = ds_transformed[0][dim].encoding
            encoding.pop('chunks',None)
            encoding.pop('preferred_chunks',None)
            ds_dims_only[dim].encoding = encoding

        # Add encoding per dimension (compressors, filters etc.)

        chunks = self._output_chunks()

        if len(chunks.keys()) > 0:
            # No specified chunks - no rechunking
            logger.info(f'Rechunking: {chunks}')
            ds_dims_only.chunk(chunks)

        empty_shape = [len(v) for v in all_dims.values()]
        dask_chunks = tuple(chunks[d] if d in chunks else len(all_dims[d]) for d in all_dims.keys() )
        empty_var = da.empty(empty_shape, chunks=dask_chunks)

        for coord in default_coords:
            logger.info(f'Filling in {coord}')
            ds_dims_only = ds_dims_only.assign_coords(
                **{coord: (
                    ds_transformed[0][coord].dims, 
                    np.array(ds_transformed[0][coord]))
                }
            )
            ds_dims_only[coord].attrs = ds_transformed[0][coord].attrs

        encoding = {}
        for dsv in ds_transformed:
            var = dsv.name
            logger.info(f'Writing empty DataArray: {var}')
            ds_dims_only[var] = xr.DataArray(empty_var, dims=list(all_dims.keys()))
            ds_dims_only[var].attrs = dsv.attrs

            # Force Zarr to use Dask chunk structure
            # Preserve non-chunk encoding attributes
            if len(chunks.keys()) > 1:
                encoding[var] = {'chunks': [v for v in chunks.values()]}

        if 'batch_dim' in ds_dims_only.dims:
            ds_dims_only = ds_dims_only.reset_index('batch_dim')

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
            memory_limit: str,
            dim_spec: Union[dict,None] = None,
        ) -> dict:
        """
        Arrange the region selection information.
        
        This includes the region info for all dimensions, source chunks and memory
        limit for each worker."""

        return {
            'dataset':{
                'uri': self.uri,
                'engine': self.engine,
                'kwargs':{},
                'zarr_cache': zarr_store
            },
            'common':{'pre_transforms': self._determine_regional_transforms()},
            'variables': self.variables,
            'region_info': {'dims':dim_spec, 'region_isel':dim_spec},
            'source_chunks': self.source_chunks,
            'output_chunks': self._output_chunks(),
            'memory_limit': memory_limit
        }

    def _override_global_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Copy saved global attributes into new dataset, plus added attributes.
        """

        # Copy global attributes
        if self.global_attrs is not None:
            ds.attrs = self.global_attrs

        if self.add_attrs is not None:
            for k,v in self.add_attrs.items():
                ds.attrs[k] = v

        return ds

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
            overwrite: bool = True,
            recommend_changes: bool = True,
            resume: bool = False,
        ):
        """
        Method to cache selected data to a zarr store
        
        Will send out parallel workers and has option to wait for completion.

        :param generate_stats: bool. Not currently implemented. Option to generate stats on the caching process, such as time taken, memory used etc.
        """

        if not isinstance(cache_store, str):
            cache_store = str(cache_store)

        if deploy_mode == 'series':
            logger.info('Writing unparallelised dataset')
            self._transform_ds()
            for ds in self._ds:

                # Specify output chunks, although this will not rechunk directly.
                ds.chunk(self._output_chunks())

                if isinstance(ds, xr.DataArray):
                    ds = xr.Dataset({ds.name:ds})

                ds = self._override_global_attrs(ds)

                ds.to_zarr(
                    cache_store, 
                    compute=True,
                    zarr_format=2, 
                    consolidated=True,
                    write_empty_chunks=True,
                    mode='w')
                
            return

        if num_jobs is None:
            num_jobs = simultaneous_worker_limit 

        cache_dir = '/'.join(cache_store.split('/')[:-1])
        zarr_store = cache_store.split('/')[-1]
        worker_config_file = f'{cache_dir}/temp/{zarr_store}.temp.json'

        if not resume:

            # Handle overwriting existing store
            if os.path.isdir(cache_store):
                if overwrite:
                    os.system(f'rm -rf {cache_store}')
                else:
                    raise ValueError()

            if not os.path.isdir(f'{cache_dir}/temp'):
                os.makedirs(f'{cache_dir}/temp')

            if self.reconfigure:
                self._recommend_tiling(self._native_ds())

            # Perform transformations
            self._transform_ds()

            if recommend_changes:
                self._display_recommendations()

            actual_workers = self._determine_worker_arrangements(num_jobs)

            if self.reconfigure:
                # Tiled datasets
                worker_config = self._arrange_region_selector(zarr_store=cache_store, memory_limit=memory_limit)
                worker_config['region_info'], actual_workers = self._reconfigure_regions(num_jobs, memory_limit=memory_limit)
                worker_config['region_info']['region_isel'] = self.dim_spec
            else:
                worker_config = self._arrange_region_selector(zarr_store=cache_store, dim_spec=self.dim_spec, memory_limit=memory_limit)

            if actual_workers != num_jobs:
                logger.info(f'Requested job split: {num_jobs} jobs')
                logger.info(f'Actual job split: {actual_workers} jobs (Chunk/Tile limitations)')

            chunks = self._output_chunks()

            with open(worker_config_file,'w') as f:
                json.dump(worker_config, f)

            self._create_empty_zarr(worker_config)

        else:
            # Resume existing workflow
            actual_workers = simultaneous_worker_limit
            logger.info(f'Resuming with {actual_workers} workers')

            if not os.path.isfile(worker_config_file):
                raise ValueError('No worker config file found for resuming.')

        match deploy_mode:
            case 'SLURM':

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
                
                # Cluster workers set via limit
                # Number of jobs is now number of traditional workers
                status = configure_dask_deployment(
                    num_dask_workers=simultaneous_worker_limit,
                    job_ids=actual_workers,
                    worker_config_file=worker_config_file,
                    memory_limit=memory_limit,
                    threads_per_worker=1
                )

        if not status:
            raise ValueError('Caching status unsuccessful. Check worker logs for more details.')
