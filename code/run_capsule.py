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

    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
            
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(utils.get_datacube_dir() / 'session_table.parquet')
    session_table['issues'] = session_table['issues'].astype(str)
    session_ids: list[str] = session_table.query(args.session_table_query)['session_id'].values.tolist()
    logger.info(f"Got {len(session_ids)} session_ids from filtered session table")
    
    if args.session_id is not None:
        if args.session_id not in session_ids:
            logger.warning(f"{args.session_id!r} not in filtered session_ids: exiting")
            exit()
        logger.info(f"Using single session_id {args.session_id} provided via command line argument")
        session_ids = [args.session_id]
    elif utils.is_pipeline(): 
        # only one nwb will be available 
        session_ids = set(session_ids) & set(p.stem for p in utils.get_nwb_paths())
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids after filtering")

    with concurrent.futures.ProcessPoolExecutor(max_workers=None, mp_context=multiprocessing.get_context('spawn')) as executor:
        dfs = []
        futures = []
        for session_id in session_ids:
            futures.append(
                executor.submit(
                    utils.get_per_trial_spike_times, 
                    starts=(pl.col('stim_start_time') - 2, pl.col('stim_start_time'), ),
                    ends=(pl.col('stim_start_time'), pl.col('stim_start_time') + 3, ),
                    col_names=('baseline', 'response', ),
                    session_id=session_id,
                    as_counts=True,
                    keep_only_necessary_cols=True,
                )
            )
            if args.test:
                break
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit='session'):
            result = future.result() # may raise here
            dfs.append(
                result
                .select('trial_index', 'unit_id', 'baseline', 'response')
            )
    if len(dfs) == 0:
        logger.warning('No results returned, with no errors: list of sessions may be empty')
        return
    df = pl.concat(dfs)
    print(df.describe())
    path = '/results/spike_counts.parquet'
    logger.info(f"Writing results to {path}")
    df.write_parquet(path) #, compression_level=22)
    return
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
            baseline_r=pl.corr('start_time', 'baseline'),
            response_r=pl.corr('start_time', 'response'),
        )
        .group_by('unit_id')
        .agg(
            vis_baseline_r=pl.col('baseline_r').filter(pl.col('context_name')=='vis').first(),
            aud_baseline_r=pl.col('baseline_r').filter(pl.col('context_name')=='aud').first(),
            vis_response_r=pl.col('response_r').filter(pl.col('context_name')=='vis').first(),
            aud_response_r=pl.col('response_r').filter(pl.col('context_name')=='aud').first(),
            vis_baseline_r2=pl.col('baseline_r').filter(pl.col('context_name')=='vis').first().pow(2),
            aud_baseline_r2=pl.col('baseline_r').filter(pl.col('context_name')=='aud').first().pow(2),
            vis_response_r2=pl.col('response_r').filter(pl.col('context_name')=='vis').first().pow(2),
            aud_response_r2=pl.col('response_r').filter(pl.col('context_name')=='aud').first().pow(2),
        )
        .collect(streaming=True)
    )
    print(corr_df.describe())
    corr_df.write_parquet('/results/corr_values.parquet')
    
    return 
    
    # ------------------------------------------------------------------------------------------------
    # calculate anova stat for each unit
    anova_df = (
        df
        .drop_nulls()
        .drop_nans()
        .with_columns(
            session_id=pl.col('unit_id').str.split('_').list.slice(0, 2).list.join('_')
        )
        # get trial metadata ----------------------------------------------- #
        .join(
            other=(
                utils.get_df('trials')
                .select('session_id', 'trial_index', 'block_index', 'context_name', 'start_time')
            ),
            on=['session_id', 'trial_index',],
            how='inner',
        )
        # ------------------------------------------------------------------ #
        .group_by("unit_id")
        .agg(
            *[
                pl.map_groups(
                    exprs=['block_index', 'context_name', interval_name],
                    function=functools.partial(fit_anova, interval_name=interval_name),
                    return_dtype=pl.Struct,
                ).alias(f"anova_{interval_name}")
                for interval_name in ['baseline', 'response']
            ]
        )
        .unnest('anova_baseline', 'anova_response')
        .fill_nan(None)
    )
    print(anova_df.describe())
    anova_df.write_parquet('/results/anova_values.parquet')

def fit_anova(list_of_series: Sequence[pl.Series], interval_name: str) -> pl.Series:
        if len(list_of_series) != 3:
            raise ValueError(f"Expected 3 series, got {len(list_of_series)}")
        assert all(s.len() == list_of_series[0].len() for s in list_of_series)
        data = (
            pl.DataFrame(
                {'block': list_of_series[0], 'context': list_of_series[1], 'y': list_of_series[2]}
            )
            .drop_nans()
            .drop_nulls() 
        )
        stats = ('PR(>F)', 'F')
        factors = ('context', 'block')
        results = {f'{interval_name}_{stat}_{factor}': np.nan for stat in stats for factor in factors}
        if len(data) <= 2:
            return results
        if len(data['context'].unique()) == 1:
            return results
        if len(data['block'].unique()) == 1:
            return results
        model = ols('y ~ context + block', data=data.to_pandas()).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        results = {f'{interval_name}_{stat}_{factor}': anova_table.loc[factor, stat] for stat in stats for factor in factors}
        return results

if __name__ == "__main__":
    main()
