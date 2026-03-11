from zarr_parallel import ZarrParallelAssembler
from zarr_parallel.utils import set_verbose

import os

def main():
    """
    Basic example test
    """

    set_verbose(1)
    os.environ['ZP_LOG_LEVEL'] = '1'
    
    zp = ZarrParallelAssembler(
        data_uri="https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
        preprocessors = [
            {"type": "reverse_axis", "dim": "latitude"},
            #{"type": "roll", "dim": "longitude", "shifts": None}, # Roll required BEFORE subsetting
            {"type": "sel", "time": ["2000-03-02 00:00:00", "2005-01-10 23:00:00"], "latitude": [60, 67.8], "longitude": [10, 137.8]},
            # xarray-based transformations SHOULDN'T affect the region arrangements.
        ],
        chunks={"time": 48},
        engine='kerchunk',
        variables={'d2m':{}},
        cache_label='_vtest'
    )
    
    zp.cache(cache_store='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache/v0.3.2.zarr',deploy_mode='dask_distributed',simultaneous_worker_limit=4)

if __name__ == '__main__':
    main()
