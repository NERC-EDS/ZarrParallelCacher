__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

from typing import Union
import os

SLURM_CONFIG = """#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=cedaproc
#SBATCH --qos=standard
#SBATCH --job-name=EDS_AI_CACHE_JAS
#SBATCH --time=
#SBATCH --mem=
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
""".split('\n')

def configure_slurm_deployment(
        cache_dir: str,
        store_name: str,
        worker_config_file: str,
        actual_workers: int,
        simultaneous_worker_limit: Union[int,None],
        venvpath: Union[str,None] = None,
        worker_timeout: str = '30:00',
        memory_limit: str = '2GB',
        await_completion: bool = False,
):

    slurm_config_file = f'{cache_dir}/temp/{store_name}.sbatch'

    VENVPATH     = venvpath or os.environ.get('VIRTUAL_ENV')
    slurm_config = list(SLURM_CONFIG)

    slurm_config[5] += worker_timeout
    slurm_config[6] += memory_limit

    #slurm_config.append(f'source {VENVPATH}/bin/activate')
    slurm_config.append('module load jaspy')

    slurm_config.append(f'python /home/users/dwest77/cedadev/AI/FRAME-FM/tests/write_region.py {worker_config_file} $SLURM_ARRAY_TASK_ID')

    with open(slurm_config_file,'w') as f:
        f.write('\n'.join(slurm_config))

    # Dryrun mode logs the slurm file and command
    if simultaneous_worker_limit is not None:
        os.system(f'sbatch --array=0-{actual_workers-1}%{simultaneous_worker_limit} {slurm_config_file}')
    else:
        os.system(f'sbatch --array=0-{actual_workers-1} {slurm_config_file}')

    if await_completion:
        # Needs to know location of 'out' files only.
        #await_completion(worker_config_file)
        raise NotImplementedError
    
    return True