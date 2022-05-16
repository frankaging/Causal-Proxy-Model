import os
import datetime as dt

import numpy as np


def save_output(output_dir, filename, final_ATE, final_CEBaB_metrics, final_CEBaB_per_aspect_direction, final_CaCE_per_aspect_direction, final_ACaCE_per_aspect):
    # timestamp the output dir
    PST_timezone = dt.timezone(dt.timedelta(hours=-8))
    timestamp = str(dt.datetime.now(tz=PST_timezone)).split('.')[0].replace(' ','_')
    output_dir = output_dir + '_' + timestamp
    
    os.makedirs(f'{output_dir}/csv/') if not \
        os.path.isdir(f'{output_dir}/csv/') else None
    os.makedirs(f'{output_dir}/tex/') if not \
        os.path.isdir(f'{output_dir}/tex/') else None

    # save csv
    final_ATE.to_csv(f'{output_dir}/csv/ATE__{filename}.csv')
    final_CEBaB_metrics.to_csv(f'{output_dir}/csv/CEBaB__{filename}.csv')
    final_CEBaB_per_aspect_direction.to_csv(f'{output_dir}/csv/CEBaB_per_direction__{filename}.csv')
    final_CaCE_per_aspect_direction.to_csv(f'{output_dir}/csv/CaCE__{filename}.csv')
    final_ACaCE_per_aspect.to_csv(f'{output_dir}/csv/ACaCE__{filename}.csv')
    # save latex
    final_ATE.to_latex(f'{output_dir}/tex/ATE__{filename}.tex')
    final_CEBaB_metrics.to_latex(f'{output_dir}/tex/CEBaB__{filename}.tex')
    final_CEBaB_per_aspect_direction.to_latex(f'{output_dir}/tex/CEBaB_per_direction__{filename}.tex')
    final_CaCE_per_aspect_direction.to_latex(f'{output_dir}/tex/CaCE__{filename}.tex')
    final_ACaCE_per_aspect.to_latex(f'{output_dir}/tex/ACaCE__{filename}.tex')


def get_df_with_variances(dfs, contains_arrays=False):
    avg = sum(dfs) / len(dfs)
    var = sum([(df - avg) ** 2 for df in dfs]) / len(dfs)
    std = var ** (1 / 2)

    # get a function to round the dataset
    round_decimals = 2
    if contains_arrays:
        round_function = lambda df: df.applymap(lambda x: np.round(x, decimals=round_decimals))
    else:
        round_function = lambda df: df.round(round_decimals)
    return round_function(avg).astype(str) + ' (± ' + round_function(std).astype(str) + ')'


def average_over_seeds(pipeline_outputs):
    ATE = pipeline_outputs[0][0]
    CEBaB_metric = [output[1] for output in pipeline_outputs]
    CEBaB_per_aspect_direction = [output[2] for output in pipeline_outputs]
    CaCE_per_aspect_direction = [output[3] for output in pipeline_outputs]
    ACaCE_per_aspect = [output[4] for output in pipeline_outputs]

    # calculate averages and std
    CEBaB_metric = get_df_with_variances(CEBaB_metric)
    CEBaB_per_aspect_direction = get_df_with_variances(CEBaB_per_aspect_direction)
    CaCE_per_aspect_direction = get_df_with_variances(CaCE_per_aspect_direction, contains_arrays=True)
    ACaCE_per_aspect = get_df_with_variances(ACaCE_per_aspect, contains_arrays=True)

    return ATE, CEBaB_metric, CEBaB_per_aspect_direction, CaCE_per_aspect_direction, ACaCE_per_aspect
