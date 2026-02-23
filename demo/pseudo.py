from zarr_parallel import Dataset
from FRAME_FM.generic_data_loader import BigGeoDataset

ds = Dataset(croissant = 'path/to/croissant')

ds.roll(dim='longitude', shift=None)
ds.reverse_axis(dim='latitude')

ds.sel(
    time=slice("2000-03-02 00:00:00", "2010-01-10 23:00:00"), 
    latitude=slice(60, -67.8),
    longitude=slice(10, 137.8)
)

ds.rename('t2m','surface_temperature')
ds.rename('d2m','dewpoint_temperature')

geo_ds = BigGeoDataset(
    selectors=[ds],
    cache_dir='',
    # remove pre-transforms, now handled within each selector.
)