import xarray as xr
import os

from dask.distributed import Client, LocalCluster

if __name__ == '__main__':

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

    memory_limit = "2000MB"

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,#int(cpus),
        memory_limit=memory_limit,
    )

    client = Client(cluster)

    ds = xr.open_dataset('https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json', chunks='auto')

    # Breaks this node if there is no protection for downloading too much data.
    ds2 = ds['d2m'].isel(time=slice(0,25000))
    print(ds2)
    x=input()
    print(ds2.mean().compute())