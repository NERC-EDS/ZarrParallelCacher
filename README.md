# Zarr Parallel Cacher

![Current Git Release](https://img.shields.io/github/v/release/NERC-EDS/ZarrParallelCacher)
[![PyPI version](https://badge.fury.io/py/zarr-parallel.svg)](https://pypi.python.org/pypi/zarr-parallel/)

This package has been developed as part of the NERC EDS FRAME-FM AI project. It has been separated into its own module for ease of reusability across multiple projects. AI-specific steps may form part of the package, but may also be disabled by default.

See the [main documentation page](https://nerc-eds.github.io/ZarrParallelCacher/) for more details.

## Basic Usage

```
from zarr_parallel import ZarrParallelAssembler

zp = ZarrParallelAssembler(data_uri=uri, preprocessors=preprocessors,
            chunks=chunks,
            engine='kerchunk',
            variables={'d2m':{}}, 
            cache_label='_v1')

zp.cache(
    cache_dir='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache',
    deploy_mode='dask_distributed',
    simultaneous_worker_limit=4)
```

The above code snippet demonstrates the use of this package. The `data_uri` and `engine` parameters refer to the xarray `open_dataset` method for accessing the source object. `chunks` are required to specify the output chunking in the zarr cache, which is also required for organising the parallel jobs. `variables` is optional to add, and includes the ability to run transforms on specific data arrays (such as renaming) which are applied individually.

The `preprocessors` list defines the set of preprocessing transforms to apply to the dataset (including selection) at the point of caching. This should include all transforms that should be applied to the dataset before writing to the zarr cache.

The `num_jobs` and `simultaneous_worker_limit` parameters are used to configure for parallel deployment. If no `num_jobs` is provided, the assembler will calculate the optimal number of jobs for your memory limit (recommended). The default memory limit is 2GB and the timeout is set at 30 minutes, although this only applies to SLURM deployments at present.

## Transforms/Preprocessors
Transformations to the data may be specified via the selector option passed in the above example. Xarray-native transformations are supported, as well as transforms from the FRAME-FM package if installed.

## Selection Recommendations
The assembler will halt to recommend alternative data selections based on the underlying chunk structure. Proceeding without recommendations is not advised, as mismatched chunk-region borders may involve duplicating chunk requests and significantly increasing memory requirements per worker.

### Version 0.3 Changes
- Heartbeats between jobs in the dask workers.
- Now able to shut off dask distributed info messages.
- Added ability to add attributes

### Version 0.4 Changes
- Job parallelisation now distributed to workers for efficiency
    - Small parallel writes were found to be inefficient, so the writes are parallelised to the largest possible selection while adhering to memory/timeout limits.
- Tiling parallelisation now available. Caveats:
    - Tiling necessitates rechunking to single chunk-per-tile. This means tile size may need to be smaller than expected to account for memory limitations of individual worker - specifically where source chunking scheme inflates the size of data initially retrieved. Error will be raised if the estimated memory requirement per tile is larger than the memory limit for the worker.

### Version 0.5 Changes
- Fixed bugs with chunk identification for both tiled and non-tiled datasets.
- Attributes now set for parallel and series writes to zarr.
- Logging now enabled in the assembler directly - pass `log_level` argument as int from 0 to 2 for warnings/info/debugging.
- Documentation added using Mkdocs!