__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

from dask.distributed import Client, LocalCluster, get_worker, WorkerPlugin
from typing import Union
import logging
from zarr_parallel.utils import logstream

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
    print(f'Worker {dask_worker.id} processing job {job_id}')
    return True

def get_id(dask_worker):
    return dask_worker.id

def set_jobs(dask_worker, job_list):
    dask_worker.my_jobs = job_list

def configure_dask_deployment(
        num_dask_workers: int,
        job_ids: Union[int,list],
        memory_limit: str = '2GB',
        threads_per_worker: int = 1):

    cluster = LocalCluster(
        n_workers=num_dask_workers,
        threads_per_worker=threads_per_worker,#int(cpus),
        memory_limit=memory_limit,
    )

    client = Client(cluster)
    client.register_worker_callbacks(setup)

    if isinstance(job_ids, int):
        job_ids = list(range(num_job_ids))

    futures = client.scatter(job_ids, broadcast=False)
    results = client.map(process_job, futures)

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
        print('Awaiting all results:', len(results)-msgs)

    # Log the below
    print("\n--- Summary ---")
    print(f'Results: {len(results)}')
    success, failed = 0,0
    for msg in results:
        if msg.result():
            success += 1
        else:
            failed += 1
    print(f' > Success: {success}')
    print(f' > Failed: {failed}')

    return True