# Zarr Parallel Cacher

This package has been developed as part of the FRAME-FM AI project. It has been separated into its own module for ease of reusability across multiple projects. AI-specific steps may form part of the package, but may also be disabled by default.

## Basic Usage

```
from zarr_parallel.assembler import ZarrParallelAssembler

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

The `num_jobs` and `simultaneous_worker_limit` parameters are used to configure for parallel deployment. If no `num_jobs` is provided, the assembler will calculate the optimal number of jobs for your memory limit (recommended).

## Transforms/Preprocessors
Transformations to the data may be specified via the selector option passed in the above example. Xarray-native transformations are supported, as well as transforms from the FRAME-FM package if installed.

## Selection Recommendations
The assembler will halt to recommend alternative data selections based on the underlying chunk structure. Proceeding without recommendations is not advised, as mismatched chunk-region borders may involve duplicating chunk requests and significantly increasing memory requirements per worker.