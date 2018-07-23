# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import combinations
import os.path
import pkg_resources
from random import choice
import uuid

import numpy as np
from numpy.linalg.linalg import LinAlgError
from scipy import linalg
from scipy.stats import (kruskal, mannwhitneyu, wilcoxon, ttest_ind, ttest_rel,
                         ttest_1samp, f_oneway)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.formula.api import mixedlm
from skbio import DistanceMatrix
from skbio.stats.distance import MissingIDError
import q2templates


TEMPLATES = pkg_resources.resource_filename('q2_longitudinal', 'assets')


def _validate_input_values(df, metric, individual_id_column, group_column,
                           state_column, state_1, state_2):
    # confirm that different state values are input
    if state_1 is not None and state_1 == state_2:
        raise ValueError(
            'You have chosen the same value for state_1 and state_2. These '
            'parameters must be given different values.')
    # confirm that metric, individual, group, and state columns are in metadata
    # and do not match
    _validate_input_columns(
        df, individual_id_column, group_column, state_column, metric)
    # confirm that state_1 and state_2 exist in metadata and in each group in
    # group_column. Both checks are performed to give specific error messages.
    if state_1 is not None:
        for state in [state_1, state_2]:
            state = df[state_column].dtype.type(state)
            if state not in df[state_column].values:
                raise ValueError(
                    'State {0} not present in column {1} of metadata'.format(
                        state, state_column))
            if group_column is not None:
                for group in df[group_column].unique():
                    df1 = df[df[group_column] == group]
                    _validate_state_in_dataframe(
                        df1, state, state_column, group)
            else:
                _validate_state_in_dataframe(
                    df, state, state_column, 'any')


def _validate_state_in_dataframe(df1, state, state_column, group):
    if state not in df1[state_column].values:
        raise ValueError(
            'State {0} is not represented by any members of {1} group '
            'in metadata. Consider using a different '
            'group_column or state value.'.format(state, group))


def _validate_input_columns(df, individual_id_column, group_column,
                            state_column, metric):
    if isinstance(group_column, list):
        cols = [individual_id_column, state_column, metric] + group_column
    else:
        cols = [individual_id_column, group_column, state_column, metric]
    # confirm that variable values (column names) do not match
    cols = [c for c in cols if c is not None]
    if len(set(cols)) < len(cols):
        raise ValueError(
            "individual_id_column, group_column, state_column, and metric "
            "must all be set to unique values.")
    # confirm that individual, group, and state columns are in metadata
    for column in cols:
        if column not in df.columns:
            raise ValueError('{0} is not a column in your metadata'.format(
                             column))
    # check that more than one states exist
    if state_column is not None:
        states = df[state_column].unique()
        if len(states) < 2:
            raise ValueError("state_column must contain at least two unique "
                             "values. Values in {0}: {1}".format(
                                state_column, states))


def _get_group_pairs(df, group_value, individual_id_column='SubjectID',
                     group_column='Group', state_column='time_point',
                     state_values=['1', '2'], replicate_handling='error'):
    results = []
    errors = []
    if group_column is not None:
        group_members = df[group_column] == group_value
        group_md = df[group_members]
    else:
        group_md = df
    for individual_id in set(group_md[individual_id_column]):
        result = []
        for state_value in state_values:
            state_value = df[state_column].dtype.type(state_value)
            individual_id = df[individual_id_column].dtype.type(individual_id)
            _state = df[state_column] == state_value
            _ind = df[individual_id_column] == individual_id
            individual_at_state_idx = group_md[_state & _ind].index
            if len(individual_at_state_idx) > 1:
                errors.append(
                    "Multiple values for {0} {1} at {2} {3} ({4})".format(
                        individual_id_column, individual_id, state_column,
                        state_value,
                        ' '.join(map(str, individual_at_state_idx))))
                if replicate_handling == 'error':
                    raise ValueError(
                        'Detected replicate samples for individual ({0}) {1} '
                        'at state ({2}) {3}. Remove replicate values from '
                        'input files or set replicate_handling parameter to '
                        'select how replicates are handled.'.format(
                            individual_id_column, individual_id, state_column,
                            state_value))
                elif replicate_handling == 'random':
                    result.append(choice(individual_at_state_idx))
                elif replicate_handling == 'drop':
                    pass
            elif len(individual_at_state_idx) == 0:
                errors.append("No values for {0} {1} at {2} {3}".format(
                    individual_id_column, individual_id, state_column,
                    state_value))
                pass
            else:
                result.append(individual_at_state_idx[0])
        if len(result) == len(state_values):
            results.append(tuple(result))
    return results, errors


def _extract_distance_distribution(distance_matrix: DistanceMatrix, pairs,
                                   df, individual_id_column, group_column):
    result = []
    pairs_summary = []
    for p in pairs:
        try:
            dist = distance_matrix[p]
            result.append(dist)
            individual_id = df[individual_id_column][p[1]]
            group = df[group_column][p[1]]
            pairs_summary.append((p[1], individual_id, dist, group))
        except MissingIDError:
            pass
    pairs_summary = pd.DataFrame(
        pairs_summary, columns=['#SampleID', 'SubjectID', 'Distance', 'Group'])
    pairs_summary.set_index('#SampleID', inplace=True)
    return result, pairs_summary


def _between_subject_distance_distribution(
        distance_matrix: DistanceMatrix, pairs, metadata=None,
        group_column=None, group_values=None):
    '''Extract distance distribution between all sample pairs in distance
    matrix, if samples are not listed in pairs; optionally, also if both
    samples are members of group_column's group_values in metadata
    pd.DataFrame.
    pairs: list of tuples
        paired samples to exclude from "between" distribution
    metadata: pd.DataFrame
        sample metadata
    group_column: str
        dataframe column to search
    group_values: list
        groups to include
    '''
    results = dict()
    for i in distance_matrix.ids:
        # ignore sample if belongs to group
        if group_column and group_values:
            if metadata.loc[i][group_column] not in group_values:
                continue
        for j in distance_matrix.ids:
            # ignore sample if belongs to group
            if group_column and group_values:
                if metadata.loc[j][group_column] not in group_values:
                    continue
            # ignore sample if i = j (self match)
            if i == j:
                continue
            if ((i, j) not in pairs and
                    (j, i) not in pairs and
                    (j, i) not in results):
                try:
                    results[i, j] = distance_matrix[i, j]
                except MissingIDError:
                    pass
    return list(results.values())


def _tabulate_matrix_ids(distance_matrix):
    _ids = distance_matrix.ids
    return pd.Series(_ids, index=_ids)


def _get_pairwise_differences(df, pairs, category, individual_id_column,
                              group_column):
    result = []
    pairs_summary = []
    for pre_idx, post_idx in pairs:
        individual_id = df[individual_id_column][pre_idx]
        # determine group membership for results summary
        if group_column is not None:
            group = df[group_column][pre_idx]
        else:
            group = 'None'
        pre_value = float(df[category][pre_idx])
        post_value = float(df[category][post_idx])
        paired_difference = post_value - pre_value
        if not np.isnan(paired_difference):
            result.append(paired_difference)
            pairs_summary.append((
                post_idx, individual_id, paired_difference, group))
    pairs_summary = pd.DataFrame(
        pairs_summary,
        columns=['#SampleID', 'SubjectID', 'Difference', 'Group'])
    pairs_summary.set_index('#SampleID', inplace=True)
    return result, pairs_summary


def _compare_pairwise_differences(groups, parametric=False):
    pvals = []
    stat = 'W (wilcoxon signed-rank test)'
    for name, values in groups.items():
        try:
            if parametric:
                t, p = ttest_1samp(values, 0.0)
                stat = 't (one-sample t-test)'
            else:
                t, p = wilcoxon(values)
        except ValueError:
            # if test fails (e.g., because of zero variance), just skip
            pass
        pvals.append((name, t, p))
    result = pd.DataFrame(pvals, columns=["Group", stat, "P-value"])
    result.set_index(["Group"], inplace=True)
    result = _multiple_tests_correction(result, method='fdr_bh')
    return result


def _multiple_group_difference(groups, parametric=False):
    '''groups: list of lists of values.'''
    if parametric:
        stat, p_val = f_oneway(*groups)
        stat_name = 'F'
        test_name = 'One-way ANOVA'
    else:
        stat, p_val = kruskal(*groups, nan_policy='omit')
        stat_name = 'H'
        test_name = 'Kruskal Wallis test'
    multiple_group_test = pd.Series(
        [stat, p_val], index=[stat_name, 'P value'], name=test_name)
    return multiple_group_test


def _per_method_pairwise_stats(groups, paired=False, parametric=True):
    '''Perform mann whitney U tests between group distance distributions,
    followed by FDR correction. Returns pandas dataframe of p-values.
    groups: dict
        {group_names: [distribution of values to compare]}
    paired: bool
        Perform Wilcoxon signed rank test instead of Mann Whitney U. groups
        values must be ordered such that paired samples appear in the same
        order in each group.
    '''
    pvals = []
    combos = [a for a in combinations(groups.keys(), 2)]
    for a in combos:
        try:
            if not paired and not parametric:
                stat_name = 'Mann-Whitney U'
                u, p = mannwhitneyu(
                    groups[a[0]], groups[a[1]], alternative='two-sided')

            elif not paired and parametric:
                stat_name = 't (two-sample t-test)'
                u, p = ttest_ind(
                    groups[a[0]], groups[a[1]], nan_policy='raise')

            elif paired and not parametric:
                stat_name = 'W (wilcoxon signed-rank test)'
                u, p = wilcoxon(groups[a[0]], groups[a[1]])

            else:
                stat_name = 't (paired t-test)'
                u, p = ttest_rel(
                    groups[a[0]], groups[a[1]], nan_policy='raise')

            pvals.append((a[0], a[1], u, p))
        except ValueError:
            # if test fails (e.g., because of zero variance), just skip
            pass

    result = pd.DataFrame(
        pvals, columns=["Group A", "Group B", stat_name, "P-value"])
    result.set_index(['Group A', 'Group B'], inplace=True)
    result = _multiple_tests_correction(result, method='fdr_bh')

    return result


def _linear_effects(metadata, metric, state_column, group_columns,
                    individual_id_column, random_effects):
    # Assemble fixed_effects
    if group_columns is not None:
        fixed_effects = [state_column] + group_columns
    else:
        fixed_effects = [state_column]

    # Assemble random_effects
    if random_effects is not None:
        # fit random slope to state_column
        random_effects = "~{0}".format(" + ".join(random_effects))
    else:
        # fit random intercept by default
        random_effects = None

    # semicolon-delimited taxonomies cause an error; copy to new metric column
    # also starting numeral (e.g., in feature name) causes error:
    # https://github.com/qiime2/q2-longitudinal/issues/101
    if ';' in metric or metric[0].isdigit():
        # generate random column name but remove hyphens (patsy error)
        # and prepend word character (otherwise patsy splits strings starting
        # with numeral!)
        new_metric = 'f' + _generate_column_name(metadata).replace("-", "")
        metadata[new_metric] = metadata[metric]
        # store original metric name to report in viz later
        old_metric = metric
        metric = new_metric
    else:
        old_metric = None

    # format formula
    formula = "{0} ~ {1}".format(metric, " * ".join(fixed_effects))

    # generate model
    mlm = mixedlm(
        formula, metadata, groups=metadata[individual_id_column],
        re_formula=random_effects)

    # numpy.linalg.linalg.LinAlgError appears to raise
    # See https://github.com/qiime2/q2-longitudinal/issues/39
    try:
        model_fit = mlm.fit()
    except LinAlgError:
        raise ValueError(
            'Linear model will not compute due to singular matrix error. '
            'This may occur if input variables correlate closely or exhibit '
            'zero variance. Please check your input variables. Removing '
            'potential covariates may resolve this issue.')

    # summarize model and format prettily
    model_summary, model_results = model_fit.summary().tables
    model_summary = pd.Series(
        data=list(model_summary[1].values) + list(model_summary[3].values),
        index=list(model_summary[0].values) + list(model_summary[2].values),
        name='model summary').to_frame()

    # fix dependent variable name if it was renamed to avoid patsy error
    if old_metric is not None:
        model_summary.loc['Dependent Variable:', 'model summary'] = old_metric

    return model_summary, model_results, model_fit


def _boxplot_from_dict(groups, hue=None, y_label=None, x_label=None,
                       y_min=None, y_max=None, palette=None,
                       label_rotation=45):
    """Generate boxplot of metric (y) by groups (x), input as dict of lists of
    values.

    hue, color variables all pass directly to equivalently named
        variables in seaborn.boxplot().
    """
    x_tick_labels = [v for k, v in sorted(
        _add_sample_size_to_xtick_labels(groups).items())]
    vals = [v for k, v in sorted(groups.items())]

    ax = sns.boxplot(data=vals, hue=hue, palette=palette)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticklabels(x_tick_labels, rotation=label_rotation)
    return ax


def _add_sample_size_to_xtick_labels(groups):
    '''groups is a dict of lists of values.'''
    x_tick_labels = {k: '{0} (n={1})'.format(k, len(v))
                     for k, v in groups.items()}
    return x_tick_labels


def _regplot_subplots_from_dataframe(state_column, metric, metadata,
                                     group_by, lowess=False, ci=95,
                                     palette='Set1'):
    '''plot a single regplot for each group in group_by.'''
    if group_by is None:
        # create dummy group column in metadata so we can squeeze into a
        # plotting function that is designed to split on at least one column.
        group_by = [_generate_column_name(metadata)]
        metadata[group_by[0]] = ''

    num_groups = len(group_by)
    height = 6 * num_groups
    f, axes = plt.subplots(num_groups, figsize=(6, height), squeeze=False)
    for num, group_column in enumerate(group_by):
        ax = axes[num][0]
        sns.set_palette(palette)
        for name, group_data in metadata.groupby(group_column):
            sns.regplot(state_column, metric, data=group_data, fit_reg=True,
                        scatter_kws={"marker": ".", "s": 100}, label=name,
                        ax=ax, lowess=lowess, ci=ci)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f


def _multiple_tests_correction(df, method='fdr_bh'):
    try:
        df['FDR P-value'] = multipletests(df['P-value'], method=method)[1]
    # zero division error will occur if the P-value series is empty. Ignore.
    except ZeroDivisionError:
        pass
    return df.sort_index()


def _load_metadata(metadata):
    metadata = metadata.to_dataframe()
    metadata = metadata.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return metadata


def _add_metric_to_metadata(table, metadata, metric):
    '''find metric in metadata or derive from table and merge into metadata.'''
    metadata = _load_metadata(metadata)
    if metric not in metadata.columns:
        if table is not None and metric in table.columns:
            table_metric = pd.DataFrame(
                pd.to_numeric(table[metric], errors='ignore'))
            metadata = pd.concat(
                [metadata, table_metric], axis=1, join='inner')
        else:
            raise ValueError(
                'metric must be a valid metadata or feature table column.')
    _validate_is_numeric_column(metadata, metric)
    return metadata


def _visualize(output_dir, multiple_group_test=False, pairwise_tests=False,
               paired_difference_tests=False, plot=False, summary=False,
               errors=False, model_summary=False, model_results=False,
               raw_data=False, plot_name='Pairwise difference boxplot',
               residuals=False,
               pairwise_test_name='Pairwise group comparison tests'):

    pd.set_option('display.max_colwidth', -1)

    if summary is not False:
        summary = q2templates.df_to_html(summary.to_frame())

    if multiple_group_test is not False:
        multiple_group_test = multiple_group_test.to_frame().transpose()
        multiple_group_test = q2templates.df_to_html(multiple_group_test)

    if pairwise_tests is not False:
        pairwise_tests.to_csv(os.path.join(output_dir, 'pairwise_tests.tsv'),
                              sep='\t')
        pairwise_tests = q2templates.df_to_html(pairwise_tests)

    if raw_data is not False:
        raw_data.to_csv(os.path.join(output_dir, 'raw-data.tsv'), sep='\t')
        raw_data = True

    if paired_difference_tests is not False:
        paired_difference_tests.to_csv(os.path.join(
            output_dir, 'paired_difference_tests.tsv'), sep='\t')
        paired_difference_tests = q2templates.df_to_html(
            paired_difference_tests)

    if model_summary is not False:
        model_summary.to_csv(os.path.join(output_dir, 'model_summary.tsv'),
                             sep='\t')
        model_summary = q2templates.df_to_html(model_summary)

    if model_results is not False:
        model_results.to_csv(os.path.join(output_dir, 'model_results.tsv'),
                             sep='\t')
        model_results = q2templates.df_to_html(model_results)

    if plot is not False:
        plot.savefig(os.path.join(output_dir, 'plot.png'), bbox_inches='tight')
        plot.savefig(os.path.join(output_dir, 'plot.pdf'), bbox_inches='tight')

    if residuals is not False:
        residuals.savefig(
            os.path.join(output_dir, 'residuals.png'), bbox_inches='tight')
        residuals.savefig(
            os.path.join(output_dir, 'residuals.pdf'), bbox_inches='tight')
        plt.close('all')

    index = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'errors': errors,
        'summary': summary,
        'model_summary': model_summary,
        'model_results': model_results,
        'multiple_group_test': multiple_group_test,
        'pairwise_tests': pairwise_tests,
        'paired_difference_tests': paired_difference_tests,
        'plot': plot,
        'residuals': residuals,
        'plot_name': plot_name,
        'raw_data': raw_data,
        'pairwise_test_name': pairwise_test_name,
    })


def _stats_and_visuals(output_dir, pairs, metric, group_column,
                       state_column, state_1, state_2,
                       individual_id_column, errors, parametric, palette,
                       replicate_handling,
                       multiple_group_test=True, pairwise_tests=True,
                       paired_difference_tests=True, boxplot=True,
                       plot_name='Pairwise difference boxplot',
                       pairwise_test_name='Pairwise group comparison tests'):
    # kruskal test or ANOVA between groups
    if multiple_group_test:
        multiple_group_test = _multiple_group_difference(
            pairs.values(), parametric=parametric)

    # pairwise testing
    if pairwise_tests:
        pairwise_tests = _per_method_pairwise_stats(
            pairs, paired=False, parametric=parametric)

    # paired difference tests
    if paired_difference_tests:
        paired_difference_tests = _compare_pairwise_differences(
            pairs, parametric=parametric)

    # boxplots
    if boxplot:
        boxplot = _boxplot_from_dict(
            pairs, palette=palette, y_label=metric, x_label=group_column)
    boxplot = boxplot.get_figure()

    summary = pd.Series(
        [metric, group_column, state_column, state_1, state_2,
         individual_id_column, parametric, replicate_handling],
        index=['Metric', 'Group column', 'State column', 'State 1',
               'State 2', 'Individual ID column', 'Parametric',
               'Replicates handling'],
        name='Paired difference tests')

    _visualize(output_dir, multiple_group_test, pairwise_tests,
               paired_difference_tests, boxplot, summary=summary,
               errors=errors, plot_name=plot_name,
               pairwise_test_name=pairwise_test_name)


def _temporal_corr(table, individual_id, corr_method="kendall"):
    '''Create Temporal correlation from a feature table
    containing repeated measures samples.
    table: pd.DataFrame
        Feature table, rows are samples, columns are features
    individual_id: pd.Series
        sample ids, with the same length as table
    corr_method: str
        temporal correlation method, "kendall", "pearson", "spearman"
    '''

    table["individual_id"] = individual_id
    results = table.groupby(["individual_id"]).corr(method=corr_method)
    results = results.fillna(0)

    return results


def _temporal_distance(corr, id_set, dist_method="fro"):
    '''Calculate Distance Matrix from temporal correlation data.
    corr: pd.DataFrame
        table grouped by individual ids, this is the output from _temporal_corr
    id_set: pd.Series
        unique subject ids from individual_id with index attached
    dist_method: str
        method supported by scipy.linalg.norm parameter ord
    '''
    id_n = len(id_set)

    dist = np.zeros((id_n, id_n))
    for i, id_i in enumerate(id_set):
        for j, id_j in enumerate(id_set[:i]):
            dist[i, j] = dist[j, i] = linalg.norm(
                corr.loc[id_i] - corr.loc[id_j], ord=dist_method)
    return DistanceMatrix(dist, ids=id_set.index)


def _nmit(table, sample_md, individual_id_column, corr_method="kendall",
          dist_method="fro"):
    '''Function to perform nonparametric microbial interdependence test (nmit)
    test.
    table: pd.DataFrame
        Feature table, rows are samples, columns are features
    sample_md: pd.DataFrame
        Sample metadata
    individual_id_column: str
        Metadata column containing IDs for individual subjects
    corr_method: str
        temporal correlation method
    dist_method: str
        temporal distance method from numpy.linalg.norm, default is "fro"
    '''
    # full series of individual ids
    individual_id = sample_md[individual_id_column]

    # compile series of unique individuals with index retained
    # the goal here is to have ordered lists of ids that we eventually append
    # to distance matrix (for metadata extraction during permanova or other
    # follow-up tests/plotting), hence we do not use pd.unique()
    id_set = individual_id.drop_duplicates()

    # calculate correlation in individuals
    _corr = _temporal_corr(table, individual_id,  corr_method)

    # calculate distance between individuals
    _dist = _temporal_distance(_corr, id_set, dist_method)

    return _dist


def _first_differences(metadata, state_column, individual_id_column, metric,
                       replicate_handling='error', baseline=None,
                       distance_matrix=None):

    # let's force states to be numeric
    _validate_is_numeric_column(metadata, state_column)

    # create dummy group column in metadata so we can use downstream functions
    # that split metadata by groups without actually bothering to do so.
    group_column = _generate_column_name(metadata)
    metadata[group_column] = 'null'

    # calculate paired difference/distance distributions between each state
    pairs_summary = pd.DataFrame()
    errors = []
    states = sorted(metadata[state_column].unique())

    # if calculating static differences, validate baseline as a valid state
    if baseline is not None:
        # cast baseline to same type as states
        baseline = metadata[state_column].dtype.type(baseline)
        # validate baseline state
        if baseline not in states:
            raise ValueError(
                'baseline must be a valid state: {0} is not in {1}'.format(
                    baseline, states))
        # otherwise splice out baseline state and peg to start of states list
        # to make iterating over the list easier.
        else:
            states.remove(baseline)
            states.insert(0, baseline)

    # iterate over range of sorted states in order to compare sequential states
    for s in range(len(states) - 1):
        # define whether to calculate first-differences (dY(t) = Y(t) - Y(t-1))
        # or calculate static differences (dY(t) = Y(t) - Y(baseline))
        if baseline is not None:
            state_1 = 0
        else:
            state_1 = s
        # get pairs of samples at each sequential state
        group_pairs, error = _get_group_pairs(
            metadata, group_value='null',
            individual_id_column=individual_id_column,
            group_column=group_column, state_column=state_column,
            state_values=[states[state_1], states[s + 1]],
            replicate_handling=replicate_handling)
        # compute distance between pairs
        if distance_matrix is not None:
            pairs, pairs_summaries = _extract_distance_distribution(
                distance_matrix, group_pairs, metadata,
                individual_id_column, group_column)
        # or compute difference between pairs
        else:
            pairs, pairs_summaries = _get_pairwise_differences(
                metadata, group_pairs, metric, individual_id_column,
                group_column)
        pairs_summary = pd.concat([pairs_summary, pairs_summaries])
        errors.extend(error)

    # convert pairs_summary to series with relevant metric
    if distance_matrix is not None:
        pairs_summary = pairs_summary['Distance']
    else:
        pairs_summary = pairs_summary['Difference']

    # raise sensible error if output is empty.
    if len(pairs_summary) == 0:
        raise RuntimeError(
            'Output is empty. Either no paired samples were detected in the '
            'inputs or replicate samples were dropped. Check input files, '
            'parameters, and replicate_handling settings.')

    return pairs_summary


# borrowed from qiime2 framework
def _generate_column_name(df):
    """Generate column name that doesn't clash with current columns."""
    while True:
        name = str(uuid.uuid4())
        if name not in df.columns:
            return name


def _validate_metadata_is_superset(metadata, table):
    metadata_ids = set(metadata.index.tolist())
    table_ids = set(table.index.tolist())
    if not table_ids.issubset(metadata_ids):
        raise ValueError('Missing samples in metadata: %r' %
                         table_ids.difference(metadata_ids))


def _validate_is_numeric_column(metadata, metric):
    if np.issubdtype(metadata[metric].dtype, np.number):
        pass
    else:
        raise ValueError('{0} is not a numeric metadata column. '
                         'Please choose a metadata column containing only '
                         'numeric values.'.format(metric))


def _summarize_feature_stats(table, state_md_col):
    feature_table = pd.concat([state_md_col, table], join='inner', axis=1)
    # calculate mean at each state value
    state = state_md_col.columns[0]
    feature_state_mean = feature_table.groupby(state).mean()
    # first differences of mean per state
    first_diffs = feature_state_mean.diff().dropna()
    # sum increase and decrease across states
    feature_md = pd.DataFrame(index=table.columns)
    feature_table.index.name = 'feature id'
    feature_md['Cumulative Avg Decrease'] = first_diffs[first_diffs < 0].sum()
    feature_md['Cumulative Avg Increase'] = first_diffs[first_diffs > 0].sum()
    # collect other descriptive stats
    feature_md['Variance'] = table.var()
    feature_md['Mean'] = table.mean()
    feature_md['Median'] = table.median()
    feature_md['Standard Deviation'] = table.std()
    feature_md['CV (%)'] = (
        feature_md['Standard Deviation'] / feature_md['Mean'] * 100)
    return feature_md
