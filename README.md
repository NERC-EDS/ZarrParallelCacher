# Zarr Parallel Cacher

This package has been developed as part of the FRAME-FM AI project. It has been separated into its own module for ease of reusability across multiple projects. AI-specific steps may form part of the package, but may also be disabled by default.

## Basic Usage

```
from zarr_parallel.assembler import ZarrParallelAssembler

zp = ZarrParallelAssembler(selector=selector, cache_label='_v1')
zp.cache(
    cache_dir='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache',
    deploy_mode='dask_distributed',
    simultaneous_worker_limit=4)
```

The above code snippet demonstrates the use of this package, where the selector is a dict object that follows the following template.

```
{   
    "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
    "common": {
        "pre_transforms": [
            {"type": "reverse_axis", "dim": "latitude"},
            #{"type": "roll", "dim": "longitude", "shifts": None}, # Roll required BEFORE subsetting
            {"type": "sel", "time": ["2000-03-02 00:00:00", "2010-01-10 23:00:00"], "latitude": [60, 67.8], "longitude": [10, 137.8]},
        ],
        "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
        "chunks": {"time": 48},  # Example chunking strategy, can be adjusted as needed,
    },
    "variables": {
        "d2m": [
            {"type": "rename", "new_name_or_name_dict":"dewpoint_temperature"},
        ],
    }
}
```

The `uri` referenced in the selector is the dataset source for this caching operation, which uses the `num_jobs` and `simultaneous_worker_limit` parameters to configure for parallel deployment. If no `num_jobs` is provided, the assembler will calculate the optimal number of jobs for your memory limit (recommended).

## Transforms
Transformations to the data may be specified via the selector option passed in the above example. Xarray-native transformations are supported, as well as transforms from the FRAME-FM package if installed.

## Selection Recommendations
The assembler will halt to recommend alternative data selections based on the underlying chunk structure. Proceeding without recommendations is not advised, as mismatched chunk-region borders may involve duplicating chunk requests and significantly increasing memory requirements per worker.