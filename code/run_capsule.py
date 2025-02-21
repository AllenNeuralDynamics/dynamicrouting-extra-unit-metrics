# stdlib imports --------------------------------------------------- #
import argparse
import concurrent.futures
import dataclasses
import json
import functools
import logging
import multiprocessing
import pathlib
import time
import types
import typing
import uuid
from typing import Any, Literal, Sequence

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import tqdm
import upath
import zarr

import statsmodels.api as sm
from statsmodels.formula.api import ols, glm
import statsmodels.stats.multitest as smm

import utils

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--update_packages_from_source', type=int, default=1)
    parser.add_argument('--session_table_query', type=str, default="is_ephys & is_task & is_annotated & is_production & issues=='[]'")
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(Params):
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")
        kwargs = {}
        if isinstance(field.type, str):
            kwargs = {'type': eval(field.type)}
        else:
            kwargs = {'type': field.type}
        if kwargs['type'] in (list, tuple):
            logger.debug(f"Cannot correctly parse list-type arguments from App Builder: skipping {field.name}")
        if isinstance(field.type, str) and field.type.startswith('Literal'):
            kwargs['type'] = str
        if isinstance(kwargs['type'], (types.UnionType, typing._UnionGenericAlias)):
            kwargs['type'] = typing.get_args(kwargs['type'])[0]
            logger.info(f"setting argparse type for union type {field.name!r} ({field.type}) as first component {kwargs['type']!r}")
        parser.add_argument(f'--{field.name}', **kwargs)
    args = parser.parse_args()
    list_args = [k for k,v in vars(args).items() if type(v) in (list, tuple)]
    if list_args:
        raise NotImplementedError(f"Cannot correctly parse list-type arguments from App Builder: remove {list_args} parameter and provide values via `override_params_json` instead")
    logger.info(f"{args=}")
    return args

# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature
def process_session(session_id: str, params: "Params", test: int = 0) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    # Get nwb file
    # Currently this can fail for two reasons: 
    # - the file is missing from the datacube, or we have the path to the datacube wrong (raises a FileNotFoundError)
    # - the file is corrupted due to a bad write (raises a RecursionError)
    # Choose how to handle these as appropriate for your capsule
    try:
        nwb = utils.get_nwb(session_id, raise_on_missing=True, raise_on_bad_file=True) 
    except (FileNotFoundError, RecursionError) as exc:
        logger.info(f"Skipping {session_id}: {exc!r}")
        return
    
    # Get components from the nwb file:
    trials_df = nwb.trials[:]
    units_df = nwb.units[:]
    
    # Process data here, with test mode implemented to break out of the loop early:
    logger.info(f"Processing {session_id} with {params.to_json()}")
    results = {}
    for structure, structure_df in units_df.groupby('structure'):
        results[structure] = len(structure_df)
        if test:
            logger.info("TEST | Exiting after first structure")
            break

    # Save data to files in /results
    # If the same name is used across parallel runs of this capsule in a pipeline, a name clash will
    # occur and the pipeline will fail, so use session_id as filename prefix:
    #   /results/<sessionId>.suffix
    logger.info(f"Writing results for {session_id}")
    np.savez(f'/results/{session_id}.npz', **results)
    params.write_json(f'/results/{session_id}.json')

# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property fields (like `bins` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing.

# - if needed, we can get parameters from the command line (like `nUnitSamples` below) and pass them
#   to the dataclass (see `main()` below)

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class Params:
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        """dict of field name: value pairs, including values from property getters"""
        return dataclasses.asdict(self) | {k: getattr(self, k) for k in dir(self.__class__) if isinstance(getattr(self.__class__, k), property)}

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()
    
    utils.setup_logging()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:
    session_ids = [p.stem.remove('_units_ks4') for p in pathlib.Path('/data').rglob('*_units_ks4.parquet')]
    # session_ids = pl.scan_parquet('/data/ks4/ks4_units.parquet').select('session_id').collect()['session_id']
    with concurrent.futures.ProcessPoolExecutor(max_workers=None, mp_context=multiprocessing.get_context('spawn')) as executor:
        results = []
        futures = []
        for session_id in session_ids:
            futures.append(executor.submit(utils.get_spike_counts_by_trial, session_id))    
            if args.test:
                break
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit='sessions'):
            results.append(future.result())

    if not results:
        logger.warning('No results returned, with no errors: list of sessions may be empty')
        return

    path = '/results/spike_counts.parquet'
    logger.info(f"Writing results to {path}")
    df = pl.concat(results, how='vertical_relaxed')
    df.write_parquet(path)#, compression_level=22)
    print(df.describe())

    # calculate time vs spike count correlation for each unit, in baseline and response intervals
    corr_df = (
        df.lazy()
        .with_columns(
            session_id=pl.col('unit_id').str.split('_').list.slice(0, 2).list.join('_')
        )
        .join(
            other=utils.get_df('trials').lazy().select('session_id', 'trial_index', 'context_name', 'start_time'),
            on=['session_id', 'trial_index'],
            how='left',
        )
        .group_by('unit_id', 'context_name')
        .agg(
            pl.corr('start_time', 'baseline').pow(2).alias('baseline_r2'),
            pl.corr('start_time', 'response').pow(2).alias('response_r2'),
        )
        .group_by('unit_id')
        .agg(
            vis_baseline_r2=pl.col('baseline_r2').filter(pl.col('context_name')=='vis').first(),
            aud_baseline_r2=pl.col('baseline_r2').filter(pl.col('context_name')=='aud').first(),
            vis_response_r2=pl.col('response_r2').filter(pl.col('context_name')=='vis').first(),
            aud_response_r2=pl.col('response_r2').filter(pl.col('context_name')=='aud').first(),
        )
        .collect(streaming=True)
    )
    print(corr_df.describe())
    corr_df.write_parquet('/results/corr_values.parquet')
    
    return 
    

if __name__ == "__main__":
    main()
