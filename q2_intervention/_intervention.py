# ----------------------------------------------------------------------------
# Copyright (c) 2017-, q2-intervention development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import qiime2
import pandas as pd
from skbio import DistanceMatrix

from ._utilities import (_get_group_pairs, _extract_distance_distribution,
                         _between_subject_distance_distribution, _visualize,
                         _get_pairwise_differences, _stats_and_visuals,
                         _add_metric_to_metadata, _linear_effects,
                         _regplot_subplots_from_dataframe, _load_metadata)


def pairwise_differences(output_dir: str, metadata: qiime2.Metadata,
                         group_column: str, metric: str, state_column: str,
                         state_1: str, state_2: str, individual_id_column: str,
                         parametric: bool=False, palette: str='Set1',
                         drop_duplicates: bool=True, table: pd.DataFrame=None
                         ) -> None:

    # find metric in metadata or derive from table and merge into metadata
    metadata = _add_metric_to_metadata(table, metadata, metric)

    # calculate paired difference distributions
    pairs = {}
    group_names = metadata[group_column].unique()
    for group in group_names:
        group_pairs = _get_group_pairs(
            metadata, group_value=group,
            individual_id_column=individual_id_column,
            group_column=group_column, state_column=state_column,
            state_values=[state_1, state_2],
            drop_duplicates=drop_duplicates)
        pairs[group] = _get_pairwise_differences(metadata, group_pairs, metric)

    # Calculate test statistics and generate boxplots
    _stats_and_visuals(
        output_dir, pairs, metric, group_column, state_column, state_1,
        state_2, individual_id_column, parametric, palette,
        drop_duplicates, multiple_group_test=True, pairwise_tests=True,
        paired_difference_tests=True, boxplot=True)


def pairwise_distances(output_dir: str, distance_matrix: DistanceMatrix,
                       metadata: qiime2.Metadata, group_column: str,
                       state_column: str, state_1: str, state_2: str,
                       individual_id_column: str, parametric: bool=False,
                       palette: str='Set1', drop_duplicates: bool=True,
                       between_group_distance: bool=False) -> None:

    metadata = _load_metadata(metadata)

    # calculate pairwise distance distributions
    pairs = {}
    group_names = metadata[group_column].unique()
    for group in group_names:
        group_pairs = _get_group_pairs(
            metadata, group_value=group,
            individual_id_column=individual_id_column,
            group_column=group_column, state_column=state_column,
            state_values=[state_1, state_2],
            drop_duplicates=drop_duplicates)
        pairs[group] = _extract_distance_distribution(
            distance_matrix, group_pairs)
        if between_group_distance:
            between = group + '_between_subject'
            pairs[between] = _between_subject_distance_distribution(
                distance_matrix, group_pairs, metadata, group_column, group)

    # Calculate test statistics and generate boxplots
    _stats_and_visuals(
        output_dir, pairs, 'distance', group_column,
        state_column, state_1, state_2, individual_id_column,
        parametric, palette, drop_duplicates, multiple_group_test=True,
        pairwise_tests=True, paired_difference_tests=False, boxplot=True)


def linear_mixed_effects(output_dir: str, metadata: qiime2.Metadata,
                         group_categories: str, metric: str, state_column: str,
                         individual_id_column: str, table: pd.DataFrame=None,
                         palette: str='Set1', lowess: bool=False, ci: int=95
                         ) -> None:
    # split group_categories into list of categories
    group_categories = group_categories.split(",")

    # find metric in metadata or derive from table and merge into metadata
    metadata = _add_metric_to_metadata(table, metadata, metric)

    # Generate LME model summary
    model_summary, model_results = _linear_effects(
        metadata, metric, state_column, group_categories,
        individual_id_column)

    # Plot dependent variable as function of independent variables
    g = _regplot_subplots_from_dataframe(
        state_column, metric, metadata, group_categories, lowess=lowess,
        ci=ci, palette=palette)

    # summarize parameters and visualize
    summary = pd.Series(
        [metric, ", ".join(group_categories), state_column,
         individual_id_column],
        index=['Metric', 'Group column', 'State column',
               'Individual ID column'],
        name='Linear mixed effects parameters')

    _visualize(output_dir, model_summary=model_summary,
               model_results=model_results, plot=g, summary=summary)
