__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

from dask.distributed import Client, LocalCluster, get_worker, WorkerPlugin
from typing import Union
import logging
from zarr_parallel.utils import logstream, set_verbose
from zarr_parallel.region import RegionWorker
import xarray as xr
import os
import gc
import time

logger = logging.getLogger('ZP.' + __name__)
logger.addHandler(logstream)
logger.propagate = False

class IDPlugin(WorkerPlugin):
    def setup(self, worker):
        worker.my_id = worker.id

def setup(dask_worker):
    from dask.distributed import Worker
    assert isinstance(dask_worker, Worker)

def process_jobs(job_ids):
    set_verbose(int(os.environ.get('ZP_LOG_LEVEL',0)))

    dask_worker = get_worker()
    config = os.environ.get("DASK_WORKER_CONFIG")

    for job_id in job_ids:
        logger.info(f'Worker {dask_worker.id} processing job {job_id}')
        # config defined the same for all workers

        rw = RegionWorker(job_id, config)
        rw.write_data_region()
        logger.info(f'Worker {dask_worker.id} processed job {job_id}')

        # Garbage Collect all data for this current worker.
        gc.collect()
    return True

def get_id(dask_worker):
    return dask_worker.id

def configure_dask_deployment(
        num_dask_workers: int,
        job_ids: Union[int,list],
        worker_config_file: str,
        memory_limit: str = '2GB',
        threads_per_worker: int = 1):
    
    os.environ["DASK_WORKER_CONFIG"] = worker_config_file

    cluster = LocalCluster(
        n_workers=num_dask_workers,
        threads_per_worker=threads_per_worker,#int(cpus),
        memory_limit=memory_limit,
    )

    client = Client(cluster)
    client.register_worker_callbacks(setup)

    if isinstance(job_ids, int):
        job_ids = list(range(job_ids))

    worker_id = 0
    worker_jobs = {i: [] for i in range(num_dask_workers)}
    for job_id in job_ids:
        worker_jobs[worker_id].append(job_id)
        worker_id += 1
        if worker_id >= num_dask_workers:
            worker_id=0

    futures = client.scatter(list(worker_jobs.values()), broadcast=False)
    results = client.map(process_jobs, futures)

    complete = False
    while not complete:
        msgs = 0
        is_complete = True
        for msg in results:
            if msg.result() is None:
                is_complete = False
            else:
                msgs += 1

        complete = is_complete
        print(f'Awaiting all results: {msgs}/{len(results)}')
        time.sleep(1)

    success, failed = 0,0
    for msg in results:
        if msg.result():
            success += 1
        else:
            failed += 1

    client.close()

    print("\n--- Summary ---")
    print(f'Results: {len(results)}')
    print(f' > Success: {success}')
    print(f' > Failed: {failed}')

    return True