import logging
import math
import sys

import xarray as xr
import yaml

from zarr_parallel.utils import logstream

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

import os

from dask.distributed import Client, LocalCluster

# SLURM provides memory in MB
mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")

if mem_per_node:
    memory_limit = f"{int(mem_per_node)}MB"
elif mem_per_cpu and cpus:
    memory_limit = f"{int(mem_per_cpu) * int(cpus)}MB"
else:
    memory_limit = "auto"  # fallback

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

def id_to_coord(id: int, coord_space: list):

    if id > math.prod(coord_space)-1:
        raise ValueError(f'ID {id} invalid for space with {math.prod(coord_space)} tiles')

    coords = []
    for dim in range(len(coord_space)):
        coord = 0

        # Calculate stride product
        strideprod = 1
        for stride in coord_space[dim+1:]:
            strideprod *= stride
        coord += math.floor(id/strideprod) % coord_space[dim]

        coords.append(int(coord))
    return coords

def extract_subset(dsmeta: dict, var: str, dimensions: dict, coords: list, coord_space: list) -> xr.DataArray:
    """
    Open a remote dataset and extract an xarray DataArray
    """

    ds = xr.open_dataset(
        dsmeta['uri'],
        engine=dsmeta['engine'],
        **dsmeta['kwargs']
    )

    dslice = {}
    dcount = 0
    for d, v in dimensions.items():
        dmin = v['source_min'] - v['worker_offset']
        dmax = v['source_max'] - v['worker_offset']
        
        # Offset is ignored if we are at the boundaries
        if coords[dcount] == 0:
            dmin = v['source_min']
        if coords[dcount] == coord_space[dcount]-1:
            dmax = v['source_max']

        dslice[d] = slice(dmin, dmax)

    return ds[var].isel(**dslice), dslice

def map_region(dimensions: dict):
    coord_extent = [int((v['source_max']-v['source_min'])/v['worker_size']) for v in dimensions.values()]
    region_extent = [int(v['source_max']-v['source_min']) for v in dimensions.values()]

    return coord_extent, region_extent

def region_from_coords(coords: list, coord_extent: list, dims: list, region_extent: list):

    region = {}
    for i in range(len(dims)):
        rmin = int(region_extent[i]/coord_extent[i])*coords[i]
        rmax = int(region_extent[i]/coord_extent[i])*(coords[i]+1)

        if coords[i] == coord_extent[i]-1:
            rmax = region_extent[i]
        region[dims[i]] = slice(rmin,rmax)

    return region

# python write_region.py config.yaml $ID

def write_data_region(id: str, dataset: dict, data: dict):

    # Open config file as dict
    
    # Determine coordinates of region
    # Map to slice of total dataset
    # Extract selection 

    dims          = list(data['dimensions'].keys())

    # Map the region to obtain extents
    coord_extent, region_extent   = map_region(data['dimensions'])

    # Determine current coordinates
    coords        = id_to_coord(int(id), coord_extent)

    # Determine current region
    region = region_from_coords(coords, coord_extent, dims, region_extent)

    chunks = {d: v['cache_size'] for d, v in data['dimensions'].items()}

    # Replace with logging
    logger.info(f'ID: {id}')
    logger.info(f'Coords: {coords}')
    logger.info(f'Chunks: {chunks}')

    darr, dslice = extract_subset(
        dataset,
        var=data['variable'],
        dimensions=data['dimensions'],
        coords=coords,
        coord_space=coord_extent
    )

    # Watch for memory limits.
    darr = darr.load()

    logger.info(f"Writing {data['variable']} Region: ")
    for d, v in dslice.items():
        print(f'{d} {v} -> {region[d]}')

    darr.encoding.pop('chunks')
    darr.chunk(chunks)
    
    # Slice array based on our coordinate region
    darr = darr.isel(**region).compute()

    # Write the specific region to the zarr cache
    darr.to_zarr(
        dataset['zarr_cache'], 
        zarr_format=2, 
        compute=True, 
        consolidated=True,
        region=region,
        write_empty_chunks=True,
        mode='r+')
    
    logger.info(f'Complete for {coords}')

def write_region_from_config(id: str, config_file: str):
    """
    Write region from config file.
    
    Different variables will have different regional configurations
    so are handled differently. See the top of this file for an example
    config file to run the region writer script.
    """

    with open(config_file) as stream:
        content = yaml.safe_load(stream)

    write_data_region(id, content['dataset'], content['data'])

if __name__ == '__main__':
    id = sys.argv[-1]
    config = sys.argv[-2]

    write_region_from_config(id, config)