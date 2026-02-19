# Replaces the cache_data_to_zarr function with the below code
import logging
import math
import os
from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
import yaml

from zarr_parallel.utils import logstream

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

# from .utils import SLURM_CONFIG
SLURM_CONFIG = """#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=cedaproc
#SBATCH --qos=standard
#SBATCH --job-name=EDS_AI_CACHE_JAS
#SBATCH --time=30:00
#SBATCH --mem=2G
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
""".split('\n')

def divide_workers(workers, dim_bins, dims):

    allocations = {}
    workers = workers - len(dim_bins)

    for x, weight in enumerate(dim_bins):
        allocations[dims[x]] = 1 + round(workers*weight)

    return allocations

class ZarrParallelAssembler:

    def __init__(self, selector: dict):

        # Subject to change - not based on any established scheme for selectors
        self.uri           = selector['uri']
        self.engine        = 'kerchunk' # Discuss an option for this?
        self.variables     = selector['variables']
        self.dimensions    = selector['common']['subset']
        self.transforms    = selector['common']['pre_transforms']
        self.output_chunks = selector['common'].get('chunks')

        # Derived properties
        ds = self._obtain_ds()
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

        # If the source chunks differ between variables, raise an 
        # error as this is not yet supported.

        ds.close()

    @property
    def zarr_store(self):
        return f'{self.cache_dir}/{self.uri.split("/")[-1].split(".")[0]}.zarr'

    def _obtain_ds(self) -> xr.Dataset:
        """
        Obtain the xarray dataset source object for the parent dataset
        """
        return xr.open_dataset(self.uri, engine=self.engine, chunks='auto')

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

            # Extract from dataset directly, no other method
            region_min = np.where(dim_arr ==
                ds[dim].sel(**{dim:dinf[0], 'method':'nearest'}).to_numpy()
            )[0][0]
            region_max = np.where(dim_arr ==
                ds[dim].sel(**{dim:dinf[1], 'method':'nearest'}).to_numpy()
            )[0][0]

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
            output_chunks = self.output_chunks[dim]

            min_region = math.lcm(int(output_chunks), int(source_chunks))

            # Unresolvable chunk structure means we cannot subdivide between workers for this dimension
            if min_region > total_region:
                min_region = total_region

            # Regions per worker - will resolve to 1 if min_region == total_region
            rpw = math.ceil(total_region/(min_region*num_workers_per_dim[dim]))

            # Region_size (per worker)
            worker_size = int(rpw * min_region)

            chunk_sum = 0
            while chunk_sum <= region_min:
                chunk_sum += int(source_chunks)
            chunk_sum -= int(source_chunks)

            # Difference between chunk border and region selected
            worker_offset = region_min - chunk_sum

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
    
    def _create_empty_zarr(self, worker_config: dict, var: str):

        ds = self._obtain_ds()
        worker_dims = worker_config['data']['dimensions']

        ds_dims_only = xr.Dataset({
            d: ds[d].isel(**{
                d: slice(v['source_min'],v['source_max']) 
            }) for d, v in worker_dims.items()
        })

        # Add encoding per dimension (compressors, filters etc.)

        chunks = {d: int(v['total_region']/v['cache_size']) for d, v in worker_dims.items()}

        logger.info(f'Rechunking: {chunks}')
        ds_dims_only.chunk(chunks)

        empty_shape = [v['total_region'] for v in worker_dims.values()]
        empty_var = da.empty(empty_shape, chunks=chunks)

        ds_dims_only[var] = xr.DataArray(empty_var, dims=list(worker_dims.keys()))

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

    def cache(
            self,
            num_workers: int, # Number of workers per zarr store
            cache_dir: Union[str,object], # Can be zarr store or Pathlike?
            generate_stats: bool = False,
            deploy_mode: str = 'SLURM',
            await_completion: bool = True,
            simultaneous_worker_limit: int = 50
        ):
        """
        Method to cache selected data to a zarr store
        
        Will send out parallel workers and has option to wait for completion.
        """

        dim_spec = self._determine_region_extents()

        dim_spec, actual_workers = self._determine_worker_arrangements(dim_spec, num_workers)

        logger.info(f'Requested workers: {num_workers}')
        logger.info(f'Actual workers: {actual_workers} (Chunk limitations)')

        worker_config_tmpl = {
            'dataset':{
                'uri': self.uri,
                'engine': self.engine,
                'kwargs':{},
                'zarr_cache': self.zarr_store
            }
        }

        for var, vinf in self.variables.items():
            worker_config = dict(worker_config_tmpl)
            worker_config['data'] = {
                'variable': var,
                'dimensions': dim_spec
            }

            # Create empty zarr structure
            if not os.path.isdir(worker_config['dataset']['zarr_cache']):
                # Add selector hash, see writer in repo.
                self._create_empty_zarr(worker_config, var)

            # Output worker_config somewhere central

            # Deploy
            # - array job (SLURM) with number of workers per div
            # - pre-transforms added to worker config
            #.  - adjust above chunk assembly if necessary due to transforms.

            cfg = "_".join([self.zarr_store,var,"tmp"])
            worker_config_file = f'{cache_dir}/temp/{cfg}.yaml'

            with open(worker_config_file,'w') as f:
                yaml.dump(worker_config, f)

            if deploy_mode == 'SLURM':

                slurm_config_file = f'{cache_dir}/temp/{cfg}.sbatch'

                VENVPATH     = os.environ.get('VIRTUAL_ENV')
                slurm_config = list(SLURM_CONFIG)

                #slurm_config.append(f'source {VENVPATH}/bin/activate')
                slurm_config.append('module load jaspy')

                slurm_config.append(f'python /home/users/dwest77/cedadev/AI/FRAME-FM/tests/write_region.py {worker_config_file} $SLURM_ARRAY_TASK_ID')

                with open(slurm_config_file,'w') as f:
                    f.write('\n'.join(slurm_config))

                # Dryrun mode logs the slurm file and command
                os.system(f'sbatch --array=0-{actual_workers-1}%{simultaneous_worker_limit} {slurm_config_file}')

                if await_completion:
                    # Needs to know location of 'out' files only.
                    #await_completion(worker_config_file)
                    raise NotImplementedError

if __name__ == '__main__':
    selectors= [
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
                "common": {
                    "pre_transforms": [
                        {"type": "subset", "time": ["2000-03-02 00:00:00", "2010-01-10 23:00:00"], "latitude": [60, -67.8], "longitude": [10, 137.8]},
                        {"type": "rename", "var_id": "d2m", "new_name": "dewpoint_temperature"},
                        {"type": "roll", "dim": "longitude", "shift": None}, # Roll required BEFORE subsetting
                        {"type": "reverse_axis", "dim": "latitude"}
                        # xarray-based transformations SHOULDN'T affect the region arrangements.
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48},  # Example chunking strategy, can be adjusted as needed,
                },
                "variables": {
                    "d2m": {},
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
    
    zp = ZarrParallelAssembler(selector=selectors[0])
    zp.cache(num_workers=500,cache_dir='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache',await_completion=True)