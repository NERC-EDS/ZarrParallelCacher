__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

from dask.distributed import Client, LocalCluster, get_worker, WorkerPlugin
from typing import Union
import logging
from zarr_parallel.utils import logstream
from zarr_parallel.region import RegionWorker
import xarray as xr
import os
import gc

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

class IDPlugin(WorkerPlugin):
    def setup(self, worker):
        worker.my_id = worker.id

def setup(dask_worker):
    from dask.distributed import Worker
    assert isinstance(dask_worker, Worker)

def process_job(job_id):
    dask_worker = get_worker()
    
    # config defined the same for all workers
    config = os.environ.get("DASK_WORKER_CONFIG")
    rw = RegionWorker(job_id, config)
    rw.write_data_region()
    print(f'Worker {dask_worker.id} processed job {job_id}')

    # Garbage Collect all data for this current worker.
    gc.collect()

def get_id(dask_worker):
    return dask_worker.id

def configure_dask_deployment(
        ds: xr.DataArray,
        zarr_path: str,
        num_dask_workers: int,
        #job_ids: Union[int,list],
        #worker_config: dict,
        memory_limit: str = '2GB',
        threads_per_worker: int = 1,
        chunks: dict = None):
    
    #os.environ["DASK_WORKER_CONFIG"] = worker_config

    cluster = LocalCluster(
        n_workers=num_dask_workers,
        threads_per_worker=threads_per_worker,#int(cpus),
        memory_limit=memory_limit,
    )

    client = Client(cluster)
    client.register_worker_callbacks(setup)

    #if isinstance(job_ids, int):
    #    job_ids = list(range(job_ids))

    ds.chunk(chunks)

    futures = client.submit(lambda ds: ds.to_zarr(
        zarr_path, 
        compute=True,
        zarr_format=2, 
        consolidated=True,
        write_empty_chunks=True,
        mode='w'), ds)
    
    result = futures.result()
    print("Write complete: ",result)

    futures.release()
    client.run(lambda: __import__("gc").collect())

    #results = client.map(process_job, futures)

    # complete = False
    # while not complete:
    #     msgs = 0
    #     is_complete = True
    #     for msg in results:
    #         if msg.result() is None:
    #             is_complete = False
    #         else:
    #             msgs += 1

    #     complete = is_complete
    #     print('Awaiting all results:', len(results)-msgs)

    # # Log the below
    # print("\n--- Summary ---")
    # print(f'Results: {len(results)}')
    # success, failed = 0,0
    # for msg in results:
    #     if msg.result():
    #         success += 1
    #     else:
    #         failed += 1
    # print(f' > Success: {success}')
    # print(f' > Failed: {failed}')

    return True