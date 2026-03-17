from zarr_parallel import ZarrParallelAssembler
from zarr_parallel.utils import set_verbose

import os

def main():
    """
    Basic example test
    """

    set_verbose(2)
    os.environ['ZP_LOG_LEVEL'] = '1'
    
    zp = ZarrParallelAssembler(
        data_uri="https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
        preprocessors = [
            {"type": "reverse_axis", "dim": "latitude"},
            #{"type": "roll", "dim": "longitude", "shifts": None}, # Roll required BEFORE subsetting
            {"type": "sel", "time": ["2000-03-02 00:00:00", "2000-06-10 23:00:00"], "latitude": [60, 67.8], "longitude": [10, 137.8]},
            {"type": "tiled", "time":120, "latitude": 200, "longitude": 400}
            # xarray-based transformations SHOULDN'T affect the region arrangements.
        ],
        chunks={},
        engine='kerchunk',
        variables={'d2m':{}},
        cache_label='_vtest2',
        add_attrs={'preprocessor_hash':'gajgedgsndasdgasdg'}
    )
    
    zp.cache(cache_store='/gws/ssde/j25b/eds_ai/frame-fm/data/zarr_cache/v0.4.0.zarr',deploy_mode='dask_distributed',simultaneous_worker_limit=4, num_jobs=4)

if __name__ == '__main__':
    main()
