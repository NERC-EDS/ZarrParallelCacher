import os

from zarr_parallel import ZarrParallelAssembler
from zarr_parallel.utils import set_verbose


def main():
    """
    Basic example test
    """

    set_verbose(2)
    os.environ['ZP_LOG_LEVEL'] = '1'
    
    zp = ZarrParallelAssembler(
        data_uri="https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
        # preprocessors = [
        #     {"type": "reverse_axis", "dim": "latitude"},
        #     #{"type": "roll", "dim": "longitude", "shifts": None}, # Roll required BEFORE subsetting
        #     {"type": "sel", "time": ["2000-03-02 00:00:00", "2005-01-10 23:00:00"], "latitude": [60, 67.8], "longitude": [10, 137.8]},
        #     # xarray-based transformations SHOULDN'T affect the region arrangements.
        # ],
        preprocessors = [
            {"type": "subset", "time": ("2000-01-01 00:00:00", "2000-02-28 23:00:00"), "latitude": (60, -30), "longitude": (40, 100)},
        ],
        chunks='auto',
        engine='kerchunk',
        cache_label='_vtest'
    )
    
    zp._transform_ds()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
