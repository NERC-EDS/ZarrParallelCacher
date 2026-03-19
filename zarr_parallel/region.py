__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import json
import logging
import math
import sys
from datetime import datetime
from typing import Union

import dask.array as da
import xarray as xr

from zarr_parallel.transforms import apply_transforms
from zarr_parallel.utils import interpret_mem_limit, logstream, set_verbose

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

class RegionWorker:
    def __init__(self, id: str, config: str, heartbeat_timeout: Union[int,None] = None):

        with open(config) as f:
            content = json.load(f)

        self.id = int(id)
        self.dsinfo      = content['dataset']
        self.transforms  = content['common']['pre_transforms']
        self.variables   = content['variables']
        self.region_isel = content['region_info']['region_isel']

        # Non-tiled datasets will have the same parallelisable and source dims.
        self.parallelisable_dims  = content['region_info']['dims']
        self.fine_dims   = content['region_info'].get('fine_dims',{})
        self.source_dims = list(self.region_isel.keys())

        self.tiled = list(self.parallelisable_dims.keys()) != self.source_dims

        self.source_chunks = content['source_chunks']
        self.output_chunks = content['output_chunks']
        self.memory_limit = content['memory_limit']

        self.heartbeat = heartbeat_timeout

        # Determine coordinate/region extents
        self.coord_extent, self.region_extent   = self.map_region()

        # Determine my coordinates
        self.coords = self.id_to_coord()

        # Determine my region
        self.region = self.region_from_coords()

        self.dslice = self.resolve_region()

        self._prepare_dataset()

    def map_region(self):
        coord_extent = [math.ceil((v['source_max']-v['source_min'])/v['worker_size']) for v in self.parallelisable_dims.values()]
        region_extent = [int(v['source_max']-v['source_min']) for v in self.parallelisable_dims.values()]

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
        for i, (dim, dinfo) in enumerate(self.parallelisable_dims.items()):

            rmin = dinfo['worker_size']*self.coords[i]
            rmax = dinfo['worker_size']*(self.coords[i]+1)

            if self.coords[i] == self.coord_extent[i]-1:
                rmax = self.region_extent[i]
            region[dim] = slice(rmin,rmax)

        # Add tiled fine dims in the case of additional fine dims
        for dim, dinfo in self.fine_dims.items():
            region[dim] = slice(dinfo['source_min'], dinfo['source_max'])

        return region

    def start_from(self, var: str, ndims: int, chunks: dict) -> int:
        """
        Detect past progress on writing the zarr store
        
        Limitation: Primary chunk dimension must be the first dimension.
        """

        trailing_zeros = '.'.join(["0" for x in range(ndims-1)])

        primary_dim = list(self.parallelisable_dims.keys())[0]

        zero_chunk = self.parallelisable_dims[primary_dim]['worker_size']/self.parallelisable_dims[primary_dim]['cache_size']
        chunk_id    = self.coords[0] * zero_chunk

        file = f"{self.dsinfo['zarr_cache']}/{var}/{chunk_id}." + trailing_zeros
        while os.path.isfile(file):
            file = f"{self.dsinfo['zarr_cache']}/{var}/{chunk_id}." + trailing_zeros
            logger.debug(f"Locating {self.dsinfo['zarr_cache']}/{var}/{chunk_id}." + trailing_zeros)
            chunk_id += 1

        if chunk_id > zero_chunk:
            logger.info(f"Resuming from chunk ID: {chunk_id}")
        else:
            logger.info(f"Starting from chunk ID: {chunk_id}")
        return chunk_id - zero_chunk
    
    def write_data_region(self):

        # Open config file as dict
        
        # Determine coordinates of region
        # Map to slice of total dataset
        # Extract selection 

        # Determine current region

        chunks = self.output_chunks

        # Replace with logging
        logger.info(f'ID: {self.id}')
        logger.info(f'Coords: {self.coords}')
        logger.info(f'Chunks: {chunks}')

        darrs = self.extract_subset()
        for darr in darrs:

            var = darr.name

            logger.info(f"Writing {var} Region")

            # heartbeat required - split into sections 
            # chunking required - split into sections
            start_from = self.start_from(var, len(darr.dims), chunks)

            force_rechunk = chunks != {} and chunks != self.source_chunks
            if not force_rechunk and not self.heartbeat and not start_from and not self.tiled:
                # Write the whole region to the zarr cache
                import pdb; pdb.set_trace()
                darr.to_zarr(
                    self.dsinfo['zarr_cache'], 
                    zarr_format=2, 
                    compute=True, 
                    consolidated=True,
                    region=self.region,
                    write_empty_chunks=True,
                    mode='r+')
                
            else:
                self._balanced_chunk_write(var, darr, chunks, start_from=start_from)

        logger.info(f'Complete for {self.coords}')

    def _balanced_chunk_write(self, var: str, darr: xr.DataArray, chunks: dict, start_from: int = 0):
        """
        Control the rate of chunk writes based on time/memory requirements
        """

        # Standard approach: single chunk output at a time.
        
        # Balanced approach:
        # - Chunking invokes numpy arrays - balance memory up to limit
        # - Dask workers invoke heartbeat - balance timeout up to limit
        # - Split chunk writes require max number of chunks per-write that fit within limits

        primary_dim   = list(self.parallelisable_dims.keys())[0]
        chunk_size    = chunks[primary_dim]
        prime_slice   = 0

        # Byte limit for memory
        memory_limit_bytes  = 0.85 * interpret_mem_limit(self.memory_limit)
        # Chunk limit based on memory
        max_mem_batch_chunk = math.floor(memory_limit_bytes / (math.prod(chunks.values()) * 8))
        # Offset to write region into parallel dataset
        region_write_offset = self.coords[0]*self.parallelisable_dims[primary_dim]['worker_size']

        # Limitation: Tiled datasets will always result in 1-1 tile-chunking
        if self.tiled:
            darr = darr.isel(**{
                primary_dim: slice(
                    region_write_offset,
                    region_write_offset + self.parallelisable_dims[primary_dim]['worker_size']
                )
            })
            mem_chunks = 1
            for dim, dinf in self.fine_dims.items():
                chunks[dim] = dinf['source_max'] - dinf['source_min']

                # If source chunks are larger than the tile selection, memory size is based on the source chunks
                mem_chunks *= max(self.source_chunks.get(dim.split('_')[0]), chunks[dim])

            max_mem_batch_chunk = math.floor(memory_limit_bytes / (mem_chunks * 8))

        if max_mem_batch_chunk < 1:
            raise ValueError(
                f'Memory limit too low to process even a single chunk. '
                f'Limit: {self.memory_limit}, Approx Required: {mem_chunks*8/1e6 :.2f} MB')

        chunk_batch = int(max_mem_batch_chunk)

        # Number of chunks to write
        nchunks     = math.ceil(darr[primary_dim].size/chunk_size)

        logger.info(f'Balancing chunk writes for {nchunks} chunks')

        complete = False
        while not complete:

            timings = []

            # Recalculate limits for chunk batch.
            prime_slice_lim = int(prime_slice + chunk_batch*chunk_size)
            # Handle final case + overflowing chunk batch request size
            if prime_slice_lim > darr[primary_dim].size:

                chunk_batch = int((darr[primary_dim].size - prime_slice)/chunk_size)
                prime_slice_lim = int(darr[primary_dim].size)
                complete = True

            timings = [datetime.now()]
            ds_sub = darr.isel(**{primary_dim: slice(prime_slice, prime_slice_lim)}).compute()

            # Append timing for numpy casting
            timings.append(datetime.now())

            ds_region = xr.Dataset(
                {d: ds_sub[d].to_numpy() for d in ds_sub.dims})
            
            dask_chunks = tuple([chunks[d] for d in chunks.keys()])
            ds_region[var] = xr.DataArray(da.from_array(ds_sub.to_numpy(), chunks=dask_chunks), dims=list(chunks.keys()))

            region_dict = {
                primary_dim:slice(
                    region_write_offset + prime_slice, 
                    region_write_offset + prime_slice_lim
                )
            }
            region_dict.update({d: slice(0, chunks[d]) for d in chunks.keys() if d != primary_dim})

            logger.info(
                f'Writing region '
                f'({prime_slice}, {prime_slice_lim}) -> '
                f'({region_write_offset + prime_slice}, {region_write_offset + prime_slice_lim})'
            )

            ds_region.to_zarr(
                self.dsinfo['zarr_cache'],
                compute=True,
                consolidated=True,
                zarr_format=2,
                region=region_dict,
                mode='r+',
                safe_chunks=False,
            )
            timings.append(datetime.now())

            # Next iteration, update start position
            prime_slice += chunk_batch*chunk_size

            # Increase chunk usage (if possible) or decrease as necessary
            if self.heartbeat is not None:

                max_time = max([(t - timings[0]).total_seconds() for t in timings])

                # Timeout comparison formula - allows increase in chunk batch size if timeout allows
                estm_chunk_limit = max(
                    2, int(abs(
                        ((self.heartbeat*0.85)-max_time)*(chunk_batch)/max_time
                    )
                ))/2

                if max_time < self.heartbeat*0.85:
                    batch_chunk += estm_chunk_limit
                    if batch_chunk > max_mem_batch_chunk:
                        batch_chunk = max_mem_batch_chunk
                    logger.debug(f' > Increased to {batch_chunk} chunks')
                
                if max_time >= self.heartbeat*0.85:
                    batch_chunk -= estm_chunk_limit
                    logger.debug(f' > Decreased to {batch_chunk} chunks')

        logger.info('All chunks written to zarr store')
        
    def _prepare_dataset(self):

        self.ds = xr.open_dataset(
            self.dsinfo['uri'],
            engine=self.dsinfo['engine'],
            chunks='auto',
            **self.dsinfo.get('kwargs',{})
        )

    def resolve_region(self) -> dict[slice]:
        """
        Resolve region to determine slice to subset
        
        Currently superfluous as the region is equal to the slice,
        but if the improvement to directly slice from source is made,
        this function becomes useful again."""
        
        dslice = {}
        dcount = 0

        # Tiled dataset - special case for resolving the region
        if self.tiled:
            for d, v in self.region_isel.items():
                dslice[d] = slice(v['source_min'], v['source_max'])
            return dslice

        # This creates the pre-tiled slice to apply to the dataset.
        for d, v in self.region_isel.items():
            dmin = v['source_min'] + self.coords[dcount]*v['worker_size']
            dmax = dmin + v['worker_size']
            
            # Adjust to fit boundary in the case of smaller final chunk
            if self.coords[dcount] == self.coord_extent[dcount]-1:
                dmax = v['source_max']

            dslice[d] = slice(dmin, dmax)
            dcount += 1

        return dslice

    def extract_subset(self) -> xr.DataArray:
        """
        Open a remote dataset and extract an xarray DataArray
        """

        # All selected transforms applied in correct order.
        transformed = apply_transforms(
            self.ds,
            common_transforms=self.transforms,
            variable_transforms=self.variables,
            region_transform=self.dslice
        )
        return transformed['datasets']


if __name__ == '__main__':
    id = sys.argv[-1]
    config = sys.argv[-2]

    set_verbose(2)

    rw = RegionWorker(id, config)
    rw.write_data_region()

    import os

    import psutil

    process = psutil.Process(os.getpid())
    print("Memory used (RSS):", process.memory_info().rss, "bytes")
