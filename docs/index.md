# ZarrParallelCacher

![Current Git Release](https://img.shields.io/github/v/release/NERC-EDS/ZarrParallelCacher)
[![PyPI version](https://badge.fury.io/py/zarr-parallel.svg)](https://pypi.python.org/pypi/zarr-parallel/)

This package has been developed as part of the NERC EDS FRAME-FM AI project. It has been separated into its own module for ease of reusability across multiple projects. AI-specific steps may form part of the package, but may also be disabled by default.

## Basic Usage

```
from zarr_parallel import ZarrParallelAssembler

zp = ZarrParallelAssembler(data_uri=uri, preprocessors=preprocessors,
            chunks=chunks,
            engine='kerchunk',
            cache_label='_v1')

zp.cache(
    cache_dir='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache',
    deploy_mode='dask_distributed',
    simultaneous_worker_limit=4)
```

The above code snippet demonstrates the use of this package. The `data_uri` and `engine` parameters refer to the xarray `open_dataset` method for accessing the source object. `chunks` are required to specify the output chunking in the zarr cache, which is also required for organising the parallel jobs. `variables` is optional to add, and includes the ability to run transforms on specific data arrays (such as renaming) which are applied individually.

The `preprocessors` list defines the set of preprocessing transforms to apply to the dataset (including selection) at the point of caching. This should include all transforms that should be applied to the dataset before writing to the zarr cache.

The `num_jobs` and `simultaneous_worker_limit` parameters are used to configure for parallel deployment. If no `num_jobs` is provided, the assembler will calculate the optimal number of jobs for your memory limit (recommended). The default memory limit is 2GB and the timeout is set at 30 minutes, although this only applies to SLURM deployments at present.

## Assembler

See the [assembler source code](assembler.md) to understand how the assembler calculates region/chunk information to assemble parallel jobs.

## Regions

Zarr supports writing regions of data natively, such that parallel jobs can write different regions into the zarr dataset simultaneously. The assembler takes care of splitting the dataset into non-overlapping chunk-aware regions that may also be tiled (for AI/ML purposes). See the [region source code](region.md) to understand how the region writer applies worker-level job divisions to essentially write a batch of chunks to disk. One batch consists of one or more chunks, allowing for memory limitations when re-chunking as well as heartbeat requirements between writes (i.e for dask distributed workers).

## Transforms/Preprocessors
Transformations to the data may be specified via the selector option passed in the above example. Xarray-native transformations are supported, as well as transforms from the FRAME-FM package if installed.

## Selection/Tiling Recommendations
The assembler will halt to recommend alternative data selections and tiling schemes based on the underlying chunk structure. Proceeding without recommendations is not advised, as mismatched chunk-region borders may involve duplicating chunk requests and significantly increasing memory requirements per worker.
