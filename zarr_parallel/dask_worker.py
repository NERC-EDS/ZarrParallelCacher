__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import gc
import logging
import os
import time
from typing import Union

import xarray as xr
from dask.distributed import Client, LocalCluster, WorkerPlugin, get_worker

from zarr_parallel.region import RegionWorker
from zarr_parallel.utils import logstream, set_verbose

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
    """
    Process a set of jobs per worker.
    
    Each worker now typically receives one job, and performs worker-parallelisation
    to write chunks individually within the region.
    """

    set_verbose(int(os.environ.get('ZP_LOG_LEVEL',0)))

    dask_worker = get_worker()
    loop = dask_worker.loop
    loop.add_callback(dask_worker.heartbeat)
    config = os.environ.get("DASK_WORKER_CONFIG")

    for job_id in job_ids:
        logger.info(f'Worker {dask_worker.id} processing job {job_id}')

        # Parallelised regions internally based on primary dimension
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
        threads_per_worker: int = 1) -> bool:
    """
    Configure distributed dask deployment for parallelised caching."""
    
    os.environ["DASK_WORKER_CONFIG"] = worker_config_file

    cluster = LocalCluster(
        n_workers=num_dask_workers,
        threads_per_worker=threads_per_worker,
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
        logger.info(f'Awaiting all results: {msgs}/{len(results)}')
        time.sleep(1)

    success, failed = 0,0
    for msg in results:
        if msg.result():
            success += 1
        else:
            failed += 1

    client.close()

    logger.info("\n--- Summary ---")
    logger.info(f'Results: {len(results)}')
    logger.info(f' > Success: {success}')
    logger.info(f' > Failed: {failed}')

    return True