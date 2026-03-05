__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import logging
import math
import sys

import xarray as xr
import json

from zarr_parallel.utils import logstream
from zarr_parallel.transforms import apply_transforms

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

import os

from dask.distributed import Client, LocalCluster

# SLURM provides memory in MB
mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")

task_id = os.environ.get("SLURM_ARRAY_TASK_ID",None)

if mem_per_node:
    memory_limit = f"{int(mem_per_node)}MB"
elif mem_per_cpu and cpus:
    memory_limit = f"{int(mem_per_cpu) * int(cpus)}MB"
else:
    memory_limit = "auto"  # fallback

if task_id is not None:
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=int(cpus),
        memory_limit=memory_limit,
    )

    client = Client(cluster)

# Takes a YAML config file as input that looks like this

# dataset:
#.  uri: https://...
#.  engine: kerchunk
#.  kwargs...
#.  zarr_cache: ...
# data:
#   variable: 'clt'
#.  dimensions:
#.    time:
#.      source_slice_min/max: A to B
#.      cache_chunk_size: 30
#.      worker_slice_size: 300

# - (A-B)/worker_slice_size gives
#   the size of the region space in that dimension (e.g 10x2x2 -> 40 workers/regions)
# - this script is given a task ID which is translated to a position in the region space.

class RegionWorker:
    def __init__(self, id: str, config: str):

        with open(config) as f:
            content = json.load(f)

        self.id = int(id)
        self.dsinfo      = content['dataset']
        self.transforms  = content['common']['pre_transforms']
        self.variables   = content['variables']
        self.region_info = content['region_info']

        self.dimensions = list(self.region_info.keys())

        # Determine coordinate/region extents
        self.coord_extent, self.region_extent   = self.map_region()

        # Determine my coordinates
        self.coords = self.id_to_coord()

        # Determine my region
        self.region = self.region_from_coords()

        self.dslice = self.resolve_region()

        self._prepare_dataset()

    def map_region(self):
        coord_extent = [int((v['source_max']-v['source_min'])/v['worker_size']) for v in self.region_info.values()]
        region_extent = [int(v['source_max']-v['source_min']) for v in self.region_info.values()]

        return coord_extent, region_extent

    def id_to_coord(self):

        if self.id > math.prod(self.coord_extent)-1:
            raise ValueError(f'ID {self.id} invalid for space with {math.prod(self.coord_extent)} tiles')

        coords = []
        for dim in range(len(self.coord_extent)):
            coord = 0

            # Calculate stride product
            strideprod = 1
            for stride in self.coord_extent[dim+1:]:
                strideprod *= stride
            coord += math.floor(self.id/strideprod) % self.coord_extent[dim]

            coords.append(int(coord))
        return coords
    
    def region_from_coords(self):

        region = {}
        for i in range(len(self.dimensions)):
            rmin = int(self.region_extent[i]/self.coord_extent[i])*self.coords[i]
            rmax = int(self.region_extent[i]/self.coord_extent[i])*(self.coords[i]+1)

            if self.coords[i] == self.coord_extent[i]-1:
                rmax = self.region_extent[i]
            region[self.dimensions[i]] = slice(rmin,rmax)

        return region
    
    def write_data_region(self):

        # Open config file as dict
        
        # Determine coordinates of region
        # Map to slice of total dataset
        # Extract selection 

        # Determine current region

        chunks = {d: min(v['cache_size'],v['worker_size']) for d, v in self.region_info.items()}

        # Replace with logging
        logger.info(f'ID: {self.id}')
        logger.info(f'Coords: {self.coords}')
        logger.info(f'Chunks: {chunks}')

        for var, vinfo in self.variables.items():


            darr = self.extract_subset({var:vinfo})

            # Watch for memory limits.
            darr = darr.load()

            logger.info(f"Writing {var} Region: ")
            for d, v in self.dslice.items():
                print(f'{d} {v} -> {self.region[d]}')

            darr.encoding.pop('chunks')
            darr.chunk(chunks)

            print(darr)

            # Write the specific region to the zarr cache
            darr.to_zarr(
                self.dsinfo['zarr_cache'], 
                zarr_format=2, 
                compute=True, 
                consolidated=True,
                region=self.region,
                write_empty_chunks=True,
                mode='r+')

            # logger.info(f"Writing {var} Region: ")
            # for d, v in self.dslice.items():
            #     print(f'{d} {v} -> {self.region[d]}')

            # # Force rechunk here to bypass dask existing chunks
            # darr_out = xr.Dataset({
            #     d: darr[d].to_numpy()
            #     for d in darr.dims})

            # var = darr.name
            # darr_out[var] = xr.DataArray(darr.to_numpy(), dims=darr.dims)
            # darr_out[var].attrs = darr.attrs

            # darr_out.chunk(chunks)

            # # Write the specific region to the zarr cache
            # darr_out.to_zarr(
            #     self.dsinfo['zarr_cache'], 
            #     zarr_format=2, 
            #     compute=True, 
            #     consolidated=True,
            #     region=self.region,
            #     write_empty_chunks=True,
            #     mode='r+')
        
        logger.info(f'Complete for {self.coords}')

    def _prepare_dataset(self):

        self.ds = xr.open_dataset(
            self.dsinfo['uri'],
            engine=self.dsinfo['engine'],
            chunks={},
            **self.dsinfo.get('kwargs',{})
        )

    def resolve_region(self) -> dict[slice]:
        
        dslice = {}
        dcount = 0
        for d, v in self.region_info.items():
            dmin = v['source_min'] - v['worker_offset']
            dmax = dmin + v['worker_size']
            
            # Offset is ignored if we are at the boundaries
            if self.coords[dcount] == 0:
                dmin = v['source_min']
            if self.coords[dcount] == self.coord_extent[dcount]-1:
                dmax = v['source_max']

            dslice[d] = slice(dmin, dmax)
            dcount += 1

        return dslice

    def extract_subset(self, vtransform: dict) -> xr.DataArray:
        """
        Open a remote dataset and extract an xarray DataArray
        """

        # All selected transforms applied in correct order.
        ds_transformed, _ = apply_transforms(
            self.ds,
            common_transforms=self.transforms,
            variable_transforms=vtransform,
            region_transform=self.dslice
        )
        return ds_transformed


if __name__ == '__main__':
    id = sys.argv[-1]
    config = sys.argv[-2]

    rw = RegionWorker(id, config)
    rw.write_data_region()