__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

import logging

logging.basicConfig(level=logging.WARNING)
logstream = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s [%(name)s]: %(message)s')
logstream.setFormatter(formatter)

def set_verbose(level: int, all: bool = False):
    """
    Reset the logger basic config.
    """

    levels = [
        logging.WARN,
        logging.INFO,
        logging.DEBUG,
    ]

    if level >= len(levels):
        level = len(levels) - 1

    for name in logging.root.manager.loggerDict:
        if 'ZP' in name or all:
            lg = logging.getLogger(name)
            lg.setLevel(levels[level])
            # Test
            lg.info('Initialised')

def interpret_mem_limit(memory_limit: str) -> int:

    mem_units = ['B','KB','MB','GB','TB','PB']
    bibi_units = ['z','KIB','MIB','GIB','TIB','PIB']

    suffix = memory_limit[-2:]
    if suffix[0].isnumeric():
        suffix = suffix[-1]

    mem = float(memory_limit.replace(suffix,''))
    if suffix in mem_units:
        mem*=(1000**mem_units.index(suffix.upper()))
    elif suffix in bibi_units:
        mem*=(1024**mem_units.index(suffix.upper()))
    else:
        raise ValueError(
            f'Memory Limit format unrecognised: {memory_limit} - '
            'Suffix should conform to e.g MB/GiB'
        )

    return int(mem)
