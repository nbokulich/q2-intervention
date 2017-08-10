# ----------------------------------------------------------------------------
# Copyright (c) 2017-, q2-intervention development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from qiime2.plugin import Str, Bool, Plugin, Metadata, Choices, Range, Float
from q2_types.feature_table import FeatureTable, RelativeFrequency
from ._intervention import (pairwise_differences, pairwise_distances,
                            linear_mixed_effects)
import q2_intervention
from q2_types.distance_matrix import DistanceMatrix


plugin = Plugin(
    name='intervention',
    version=q2_intervention.__version__,
    website="https://github.com/nbokulich/q2-intervention",
    package='q2_intervention',
    description=(
        'This QIIME 2 plugin supports methods for analysis of time series '
        'analysis, involving either paired sample comparisons or longitudinal '
        'study designs.'),
    short_description='Plugin for paired sample and time series analyses.'
)


base_parameters = {
    'metadata': Metadata,
    'state_column': Str,
    'individual_id_column': Str,
    'palette': Str % Choices([
        'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'tab10', 'tab20', 'tab20b', 'tab20c', 'viridis', 'plasma',
        'inferno', 'magma', 'terrain', 'rainbow']),
}

paired_params = {
    **base_parameters,
    'group_column': Str,
    'state_1': Str,
    'state_2': Str,
    'parametric': Bool,
    'drop_duplicates': Bool,
}

base_parameter_descriptions = {
        'metadata': (
            'Sample metadata containing group_column,  state_column, '
            'individual_id_column, and optionally metric values.'),
        'state_column': ('Metadata column containing state (e.g., Time) '
                         'across which samples are paired.'),
        'individual_id_column': (
            'Metadata column containing subject IDs  to use for pairing '
            'samples. WARNING: if duplicates exist for an individual ID at '
            'either state_1 or state_2, that subject will be dropped and '
            'reported in standard output by default. Set duplicates="ignore" '
            'to instead randomly select one member, and use --verbose to list '
            'conflicts.'),
        'palette': 'Color palette to use for generating boxplots.',
}

paired_parameter_descriptions = {
        **base_parameter_descriptions,
        'group_column': (
            'Metadata column on which to separate groups for comparison'),
        'state_1': 'Baseline state column value.',
        'state_2': 'State column value to pair with baseline.',
        'parametric': ('Perform parametric (ANOVA and t-tests) or non-'
                       'parametric (Kruskal-Wallis, Wilcoxon, and Mann-'
                       'Whitney U tests) statistical tests.'),
        'drop_duplicates': (
            'If True, will discard all subject IDs with duplicate samples '
            'at either state_1 or state_2. If False, will instead '
            'choose one representative at random from among duplicates.')
}


plugin.visualizers.register_function(
    function=pairwise_differences,
    inputs={'table': FeatureTable[RelativeFrequency]},
    parameters={**paired_params,
                'metric': Str},
    input_descriptions={'table': (
        'Feature table to optionally use for paired comparisons.')},
    parameter_descriptions={
        **paired_parameter_descriptions,
        'metric': 'Numerical metadata or artifact column to test.',
    },
    name='Paired difference testing and boxplots',
    description=(
        'Performs paired difference testing between samples from each '
        'subject. Sample pairs may represent a typical intervention study, '
        'e.g., samples collected pre- and post-treatment; paired samples from '
        'two different timepoints (e.g., in a longitudinal study design), or '
        'identical samples receiving different two different treatments. '
        'This action tests whether the change in a continuous metadata value '
        '"metric" differs from zero and differs between groups (e.g., groups '
        'of subjects receiving different treatments), and produces boxplots '
        'of paired difference distributions for each group. A feature table '
        'artifact is required input, though whether "metric" is derived from '
        'the feature table or metadata is optional.')
)


plugin.visualizers.register_function(
    function=pairwise_distances,
    inputs={'distance_matrix': DistanceMatrix},
    parameters={**paired_params,
                'between_group_distance': Bool},
    input_descriptions={'distance_matrix': (
        'Matrix of distances between pairs of samples.')},
    parameter_descriptions={
        **paired_parameter_descriptions,
        'between_group_distance': (
            'Whether to compare within-subject distances to distances between '
            'all subjects that share the same metadata group_column.')
    },
    name='Paired pairwise distance testing and boxplots',
    description=(
        'Performs pairwise distance testing between sample pairs from each '
        'subject. Sample pairs may represent a typical intervention study, '
        'e.g., samples collected pre- and post-treatment; paired samples from '
        'two different timepoints (e.g., in a longitudinal study design), or '
        'identical samples receiving different two different treatments. '
        'This action tests whether the pairwise distance between each subject '
        'pair differs between groups (e.g., groups of subjects receiving '
        'different treatments) and optionally (if between_group_distance is '
        'True) whether these within-subject distances are different from '
        'distances between subjects that share the same metadata '
        'group_column. Finally, produces boxplots of paired distance '
        'distributions for each group.')
)


plugin.visualizers.register_function(
    function=linear_mixed_effects,
    inputs={'table': FeatureTable[RelativeFrequency]},
    parameters={**base_parameters,
                'metric': Str,
                'group_categories': Str,
                'lowess': Bool,
                'ci': Float % Range(0, 100)},
    input_descriptions={'table': (
        'Feature table to optionally use for paired comparisons.')},
    parameter_descriptions={
        **base_parameter_descriptions,
        'metric': ('Dependent variable column name. Must be a column '
                   'name located in the metadata or feature table files.'),
        'group_categories': (
            'Comma-separated list (without spaces) of metadata categories to '
            'use as independent covariates used to determine mean structure '
            'of "metric".'),
        'lowess': ('Estimate locally weighted scatterplot smoothing. Note '
                   'that this will eliminate confidence interval plotting.'),
        'ci': 'Size of the confidence interval for the regression estimate.',
    },
    name='Linear mixed effects modeling',
    description=(
        'Linear mixed effects models evaluate the contribution of exogenous '
        'covariates "group_categories" to a single dependent variable, '
        '"metric". Perform LME and plot line plots of each group column. A '
        'feature table artifact is required input, though whether "metric" is '
        'derived from the feature table or metadata is optional.')
)
