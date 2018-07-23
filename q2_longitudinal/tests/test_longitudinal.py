# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import unittest
from io import StringIO
from warnings import filterwarnings

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import skbio
import qiime2
from qiime2.plugin.testing import TestPluginBase

from q2_longitudinal._utilities import (
    _get_group_pairs, _extract_distance_distribution,
    _get_pairwise_differences, _validate_input_values, _validate_input_columns,
    _between_subject_distance_distribution, _compare_pairwise_differences,
    _multiple_group_difference, _per_method_pairwise_stats,
    _multiple_tests_correction, _add_sample_size_to_xtick_labels,
    _temporal_corr, _temporal_distance, _nmit, _validate_is_numeric_column,
    _tabulate_matrix_ids, _validate_metadata_is_superset,
    _summarize_feature_stats)
from q2_longitudinal._longitudinal import (
    pairwise_differences, pairwise_distances, linear_mixed_effects, volatility,
    nmit, first_differences, first_distances)

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)


class TestUtilities(TestPluginBase):
    package = 'q2_longitudinal.tests'

    def test_get_group_pairs(self):
        res, err = _get_group_pairs(
            md, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2],
            replicate_handling='drop')
        self.assertEqual(res, [('0', '3'), ('1', '4'), ('2', '5')])
        # test operation without group_column
        res, err = _get_group_pairs(
            md, 'a', individual_id_column='ind', group_column=None,
            state_column='Time', state_values=[1, 2],
            replicate_handling='drop')
        self.assertEqual(res, [('0', '3'), ('1', '4'), ('2', '5'),
                               ('6', '9'), ('7', '10'), ('8', '11')])
        res, err = _get_group_pairs(
            md_dup, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2],
            replicate_handling='drop')
        self.assertEqual(res, [('0', '3')])
        res, err = _get_group_pairs(
            md_dup, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2],
            replicate_handling='random')
        self.assertEqual(res[0], ('0', '3'))
        self.assertIn(res[1], [('1', '4'), ('2', '4')])

    def test_extract_distance_distribution(self):
        res, pairs = _extract_distance_distribution(
            dm, [('0', '3'), ('2', '5')], md, 'ind', 'Group')
        self.assertAlmostEqual(res[0], 0.1)
        self.assertAlmostEqual(res[1], 0.3)

    def test_between_subject_distance_distribution(self):
        res = _between_subject_distance_distribution(
            dm, [('0', '3'), ('1', '4'), ('2', '5')], md, 'Group', 'a')
        self.assertEqual(len(res), 12)
        self.assertAlmostEqual(sorted(res)[0], 0.1)
        self.assertAlmostEqual(sorted(res)[7], 0.3)
        self.assertAlmostEqual(sorted(res)[11], 1.0)

    def test_get_pairwise_differences(self):
        res, pairs = _get_pairwise_differences(
            md, [('0', '3'), ('1', '4'), ('2', '5')], 'Value', 'ind', 'Group')
        self.assertEqual(res, [0.08, 0.06, 0.07999999999999999])

    def test_compare_pairwise_differences_parametric(self):
        res = _compare_pairwise_differences(groups, parametric=True)
        self.assertAlmostEqual(res['FDR P-value']['a'], 9.4882148564067405e-07)
        self.assertAlmostEqual(res['FDR P-value']['b'], 4.8474685173462082e-09)

    def test_compare_pairwise_differences_nonparametric(self):
        res = _compare_pairwise_differences(groups, parametric=False)
        self.assertAlmostEqual(res['FDR P-value']['a'], 0.0021830447373622506)
        self.assertAlmostEqual(res['FDR P-value']['b'], 0.0021830447373622506)

    def test_multiple_group_difference_parametric(self):
        res = _multiple_group_difference(groups.values(), parametric=True)
        self.assertAlmostEqual(res['P value'], 7.6936106994369541e-06)

    def test_multiple_group_difference_nonparametric(self):
        res = _multiple_group_difference(groups.values(), parametric=False)
        self.assertAlmostEqual(res['P value'], 6.0957929139040073e-05)

    def test_per_method_pairwise_stats_unpaired_parametric(self):
        res = _per_method_pairwise_stats(groups, paired=False, parametric=True)
        self.assertAlmostEqual(res['FDR P-value'][0], 7.693610699436966e-06)

    def test_per_method_pairwise_stats_unpaired_nonparametric(self):
        res = _per_method_pairwise_stats(
            groups, paired=False, parametric=False)
        self.assertAlmostEqual(res['FDR P-value'][0], 6.890936276106502e-05)

    def test_per_method_pairwise_stats_paired_parametric(self):
        res = _per_method_pairwise_stats(groups, paired=True, parametric=True)
        self.assertAlmostEqual(res['FDR P-value'][0], 3.085284368834677e-06)

    def test_per_method_pairwise_stats_paired_nonparametric(self):
        res = _per_method_pairwise_stats(groups, paired=True, parametric=False)
        self.assertAlmostEqual(res['FDR P-value'][0], 0.0021830447373622506)

    def test_add_sample_size_to_xtick_labels(self):
        groups = {'a': [1, 2, 3], 'b': [1, 2], 'c': [1, 2, 3]}
        labels = _add_sample_size_to_xtick_labels(groups)
        self.assertEqual(
            sorted(labels.values()), ['a (n=3)', 'b (n=2)', 'c (n=3)'])

    def test_multiple_tests_correction(self):
        test_df = pd.DataFrame(
            pd.DataFrame({'Group': [1, 2, 3], 'P-value': [1, 1, 0.01]}))
        test_df = _multiple_tests_correction(test_df)
        self.assertEqual(
            list(test_df['FDR P-value']), [1., 1., 0.030000000000000002])

    def test_multiple_tests_correction_zerodivisionerror(self):
        test_df = pd.DataFrame(
            pd.DataFrame({'Group': [], 'P-value': []}))
        test_df_mt = _multiple_tests_correction(test_df)
        # ZeroDivisionError is ignored, so new df should be empty and == old
        self.assertEqual(test_df_mt.sort_index(inplace=True),
                         test_df.sort_index(inplace=True))

    def test_temporal_corr(self):
        ind_id = pd.Series(
            [1, 2, 3, 1, 2, 3], index=['s1', 's2', 's3', 's4', 's5', 's6'])
        obs_tc = _temporal_corr(tab, ind_id)
        for k in obs_tc.keys():
            self.assertEqual(exp_tc[k].sort_index(inplace=True),
                             obs_tc[k].sort_index(inplace=True))

    def test_temporal_distance(self):
        id_set = pd.Series([1, 2, 3], index=['s1', 's2', 's3'])
        obs_td = _temporal_distance(exp_tc, id_set)
        self.assertTrue(np.array_equal(obs_td.data, exp_td))

    def test_nmit(self):
        sample_md = pd.DataFrame([1, 2, 3, 1, 2, 3], columns=['sample_id'],
                                 index=['s1', 's2', 's3', 's4', 's5', 's6'])
        obs_td = _nmit(tab, sample_md, 'sample_id')
        self.assertTrue(np.array_equal(obs_td.data, exp_td))

    def test_validate_is_numeric_column_raise_error(self):
        erroneous_metadata = pd.DataFrame({'a': [1, 2, 'b']})
        with self.assertRaisesRegex(ValueError, "is not a numeric"):
            _validate_is_numeric_column(erroneous_metadata, 'a')


# This test class really just makes sure that each plugin runs without error.
# UtilitiesTests handles all stats under the hood, so here we just want to make
# sure all plugins run smoothly.
class TestLongitudinal(TestPluginBase):
    package = 'q2_longitudinal.tests'

    def setUp(self):
        super().setUp()

        def _load_features(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(pd.DataFrame)
            return table

        def _load_dm(dm_fp):
            dm_fp = self.get_data_path(dm_fp)
            dm = qiime2.Artifact.load(dm_fp)
            dm = dm.view(skbio.DistanceMatrix)
            return dm

        self.table_ecam_fp = _load_features('ecam-table-maturity.qza')
        self.table_taxa_fp = _load_features('ecam-table-small.qza')
        self.md_ecam_fp = qiime2.Metadata.load(
            self.get_data_path('ecam_map_maturity.txt'))
        self.md_ecam_dm = _load_dm('ecam-unweighted-distance-matrix.qza')

    def test_validate_input_values(self):
        # should not raise error
        _validate_input_columns(md, "ind", "Group", "Time", None)
        _validate_input_columns(md, "ind", None, None, None)
        _validate_input_values(md, "Value", "ind", "Group", "Time", None, None)
        _validate_input_values(md, "Value", "ind", None, "Time", None, None)
        # these will raise expected errors
        with self.assertRaisesRegex(ValueError, "state_1 and state_2"):
            _validate_input_values(md, "Value", "ind", "Group", "Time", 1, 1)
        with self.assertRaisesRegex(ValueError, "not present"):
            _validate_input_values(md, "Value", "ind", "Group", "Time", 1, 3)
        with self.assertRaisesRegex(ValueError, "not a column"):
            _validate_input_values(md, "Value", "ind", "Group", "Days", 1, 2)
        with self.assertRaisesRegex(ValueError, "not a column"):
            _validate_input_columns(md, "ind", ["Group", "More stuff"], "Time",
                                    "Value")
        with self.assertRaisesRegex(ValueError, "unique values"):
            _validate_input_columns(md, "ind", "Time", "Time", "Value")
        with self.assertRaisesRegex(ValueError, "state_column must contain"):
            _validate_input_columns(
                md[md['Time'] == 1], "ind", "Group", "Time", "Value")
        dropped = md.drop(['9', '10', '11'])
        with self.assertRaisesRegex(ValueError, "not represented"):
            _validate_input_values(
                dropped, "Value", "ind", "Group", "Time", 1, 2)
        with self.assertRaisesRegex(ValueError, "state_1 and state_2"):
            pairwise_differences(
                output_dir=self.temp_dir.name, table=None,
                metadata=self.md_ecam_fp, group_column='delivery',
                state_column='month', state_1=0, state_2=0,
                individual_id_column='studyid', metric='observed_otus',
                replicate_handling='drop')
        with self.assertRaisesRegex(ValueError, "Detected replicate samples"):
            _get_group_pairs(
                md_dup, 'a', individual_id_column='ind', group_column='Group',
                state_column='Time', state_values=[1, 2],
                replicate_handling='error')

    def test_pairwise_differences(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid', metric='observed_otus',
            replicate_handling='drop')

    def test_pairwise_differences_no_group_column(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, group_column=None,
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid', metric='observed_otus',
            replicate_handling='drop')

    def test_pairwise_differences_taxa(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa',
            replicate_handling='drop')

    def test_pairwise_distances(self):
        pairwise_distances(
            output_dir=self.temp_dir.name, distance_matrix=self.md_ecam_dm,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid', replicate_handling='drop')

    def test_linear_mixed_effects(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery,diet,antiexposedall',
            individual_id_column='studyid', metric='observed_otus')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(
            self.get_data_path('linear_mixed_effects.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    def test_linear_mixed_effects_no_group_columns(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            individual_id_column='studyid', metric='observed_otus')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(
            self.get_data_path('linear_mixed_effects_no_group_columns.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    def test_linear_mixed_effects_with_random_effects(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery,diet,antiexposedall',
            random_effects='month',
            individual_id_column='studyid', metric='observed_otus')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(
            self.get_data_path('linear_mixed_effects_with_random_effects.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    def test_linear_mixed_effects_with_multiple_random_effects(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery,diet,antiexposedall',
            random_effects='month,studyid',
            individual_id_column='studyid', metric='observed_otus')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(self.get_data_path(
            'linear_mixed_effects_with_multiple_random_effects.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    def test_linear_mixed_effects_one_variable(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery',
            individual_id_column='studyid', metric='observed_otus')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(
            self.get_data_path('linear_mixed_effects_one_variable.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    def test_linear_mixed_effects_taxa(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery,diet,antiexposedall',
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa')
        obs = pd.read_csv(
            os.path.join(self.temp_dir.name, 'model_results.tsv'),
            sep='\t', index_col=0)
        exp = pd.read_csv(
            self.get_data_path('linear_mixed_effects_taxa.tsv'),
            sep='\t', index_col=0)
        pdt.assert_frame_equal(obs, exp)

    # just make sure this runs with metric name beginning with numeral
    def test_linear_mixed_effects_taxa_dodge_patsy_error(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, state_column='month',
            group_columns='delivery',
            individual_id_column='studyid',
            metric='74923f4bbde849e27fc4eda25d757e2a')

    def test_volatility(self):
        # Simultaneously "does it run" viz test
        # plus make sure spaghetti maker works
        volatility(
            output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
            state_column='month', individual_id_column='studyid')
        html_path = os.path.join(self.temp_dir.name, 'index.html')
        with open(html_path, 'r') as f:
            f = f.read()
            self.assertIn('"spaghettis"', f)
            self.assertIn('spaghettiLineThickness', f)
            self.assertIn('spaghettiLineOpacity', f)
            self.assertIn('spaghettiSymbolSize', f)
            self.assertIn('spaghettiSymbolOpacity', f)
            self.assertIn('#spaghetti-line-thickness', f)
            self.assertIn('#spaghetti-line-opacity', f)
            self.assertIn('#spaghetti-symbol-size', f)
            self.assertIn('#spaghetti-symbol-opacity', f)

    def test_volatility_no_individual_id_column(self):
        # Just a simple "does it run?" test.
        # plus make sure spaghetti maker is turned off.
        volatility(
            output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
            state_column='month')
        html_path = os.path.join(self.temp_dir.name, 'index.html')
        with open(html_path, 'r') as f:
            f = f.read()
            self.assertNotIn('"spaghettis"', f)

    def test_volatility_metric_and_group(self):
        # Just a simple "does it run?" test. Not much worth testing in terms
        # of the rendered output - vega does all the heavy lifting for us.
        volatility(
            output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
            default_metric='observed_otus', default_group_column='delivery',
            state_column='month', individual_id_column='studyid')

    def test_volatility_table(self):
        volatility(
            output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
            default_metric='e2c3ff4f647112723741aa72087f1bfa',
            default_group_column='delivery', state_column='month',
            individual_id_column='studyid', table=self.table_ecam_fp)

    def test_volatility_table_data_invalid_metric(self):
        with self.assertRaisesRegex(ValueError,
                                    "invalid_metric.*not a column"):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='invalid_metric',
                default_group_column='delivery', state_column='month',
                individual_id_column='studyid', table=self.table_ecam_fp)

    def test_volatility_must_use_unique_columns(self):
        with self.assertRaisesRegex(ValueError, "set to unique values"):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='observed_otus', default_group_column='month',
                state_column='studyid', individual_id_column='studyid')

    def test_volatility_invalid_columns(self):
        with self.assertRaisesRegex(ValueError, "'peanut' is not a column"):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='observed_otus', default_group_column='peanut',
                state_column='month', individual_id_column='studyid')

    def test_volatility_invalid_metric(self):
        with self.assertRaisesRegex(ValueError, "'peanut' is not a column"):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='peanut', default_group_column='delivery',
                state_column='month', individual_id_column='studyid')

    def test_volatility_single_state(self):
        single_state = self.md_ecam_fp.to_dataframe()
        single_state = single_state[single_state['month'] == 0]
        # state_column must contain at least two unique values...
        with self.assertRaisesRegex(ValueError, "state_column must contain"):
            volatility(
                output_dir=self.temp_dir.name,
                metadata=qiime2.Metadata(single_state),
                default_metric='observed_otus',
                default_group_column='delivery', state_column='month',
                individual_id_column='studyid')

    def test_volatility_categorical_state_column(self):
        with self.assertRaisesRegex(TypeError, 'must be numeric'):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='observed_otus',
                default_group_column='delivery', state_column='delivery',
                individual_id_column='studyid')

    def test_volatility_categorical_metric_column(self):
        with self.assertRaisesRegex(ValueError, 'delivery.*not a column'):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='delivery', default_group_column='delivery',
                state_column='month', individual_id_column='studyid')

    def test_volatility_numeric_group_column(self):
        with self.assertRaisesRegex(ValueError, 'observed_otu.*not a column'):
            volatility(
                output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
                default_metric='observed_otus',
                default_group_column='observed_otus', state_column='month',
                individual_id_column='studyid')

    def test_linear_mixed_effects_singular_matrix_error(self):
        with self.assertRaisesRegex(ValueError, "singular matrix error"):
            linear_mixed_effects(
                output_dir=self.temp_dir.name, table=None,
                metadata=self.md_ecam_fp, state_column='month',
                group_columns='diet,diet_3',
                individual_id_column='studyid', metric='observed_otus')

    def test_nmit(self):
        nmit(table=self.table_taxa_fp, metadata=self.md_ecam_fp,
             individual_id_column='studyid')

    def test_nmit_missing_table_ids(self):
        md = qiime2.Metadata(pd.DataFrame([[1]], columns=['i'],
                             index=pd.Index(['20'], name='id')))
        with self.assertRaisesRegex(ValueError, 'Missing samples'):
            nmit(table=self.table_taxa_fp, metadata=md,
                 individual_id_column='studyid')

    def test_first_differences(self):
        exp = pd.Series([0.08, 0.06, 0.07999999999999999, 0.12, 0.14,
                         0.14999999999999997],
                        index=['3', '4', '5', '9', '10', '11'],
                        name='Difference')
        exp.index.name = '#SampleID'
        obs = first_differences(
            metadata=qiime2.Metadata(md), state_column='Time',
            individual_id_column='ind',
            metric='Value', replicate_handling='drop')
        pdt.assert_series_equal(obs.sort_index(), exp.sort_index())

    # what if nothing changes between time points?
    def test_first_differences_static(self):
        exp = pd.Series([0., 0., 0., 0., 0., 0.],
                        index=['3', '4', '5', '9', '10', '11'],
                        name='Difference')
        exp.index.name = '#SampleID'
        obs = first_differences(
            metadata=qiime2.Metadata(md_static), state_column='Time',
            individual_id_column='ind',
            metric='Value', replicate_handling='drop')
        pdt.assert_series_equal(obs.sort_index(), exp.sort_index())

    def test_first_differences_drop_duplicates(self):
        obs = first_differences(
            metadata=qiime2.Metadata(md_dup), state_column='Time',
            individual_id_column='ind',
            metric='Value', replicate_handling='random')
        # The first diff of individual 2 is subject to random rep handling
        mystery_number = obs.iloc[1]
        if mystery_number < 0.051:
            self.assertAlmostEqual(mystery_number, 0.05)
        else:
            self.assertAlmostEqual(mystery_number, 0.06)

        # but other values are constant, so we will just drop in the mystery
        # value and the exp/obs series should match.
        exp = pd.Series([0.08, mystery_number, 0.12, 0.14,
                         0.14999999999999997],
                        index=['3', '4', '9', '10', '11'], name='Difference')
        exp.index.name = '#SampleID'
        pdt.assert_series_equal(obs.sort_index(), exp.sort_index())

    def test_first_differences_single_state(self):
        single_state = qiime2.Metadata(md[md['Time'] == 1])
        with self.assertRaisesRegex(ValueError, "state_column must contain"):
            first_differences(
                metadata=single_state, state_column='Time',
                individual_id_column='ind',
                metric='Value', replicate_handling='drop')

    def test_first_differences_single_individual(self):
        exp = pd.Series([0.08],
                        index=['3'],
                        name='Difference')
        exp.index.name = '#SampleID'
        single_ind = qiime2.Metadata(md[md['ind'] == 1])
        obs = first_differences(
                metadata=single_ind, state_column='Time',
                individual_id_column='ind',
                metric='Value', replicate_handling='drop')
        pdt.assert_series_equal(obs.sort_index(), exp.sort_index())

    def test_first_differences_single_sample(self):
        single_sam = qiime2.Metadata(md[(md['ind'] == 1) & (md['Time'] == 1)])
        with self.assertRaisesRegex(ValueError, "state_column must contain"):
            first_differences(
                metadata=single_sam, state_column='Time',
                individual_id_column='ind',
                metric='Value', replicate_handling='drop')

    def test_first_differences_nonnumeric_metric_error(self):
        with self.assertRaisesRegex(ValueError, "not a numeric"):
            first_differences(
                metadata=self.md_ecam_fp, state_column='month',
                individual_id_column='studyid',
                metric='delivery', replicate_handling='drop')

    def test_first_differences_taxa(self):
        exp = pd.read_csv(self.get_data_path(
            'ecam-taxa-first-differences.tsv'),
            sep='\t', squeeze=True, index_col=0)
        obs = first_differences(
            metadata=self.md_ecam_fp, state_column='month',
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa',
            replicate_handling='drop', table=self.table_ecam_fp)
        pdt.assert_series_equal(obs, exp)

    def test_first_differences_baseline(self):
        exp = pd.Series(
            [-0.01, 0., 0.01, 0.07, 0.06, 0.09, 0.07, 0.1, 0.15, 0.12, 0.16],
            index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
            name='Difference')
        exp.index.name = '#SampleID'
        obs = first_differences(
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            metric='Value', replicate_handling='drop', baseline=0)
        pdt.assert_series_equal(obs, exp)

    def test_first_differences_baseline_is_not_state_0(self):
        exp = pd.Series(
            [-0.01, -0.02, -0.01, 0.06, 0.05, 0.08, 0.06, 0.09, 0.14, 0.11,
             0.15],
            index=['0', '1', '2', '4', '5', '6', '7', '8', '9', '10', '11'],
            name='Difference')
        exp.index.name = '#SampleID'
        obs = first_differences(
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            metric='Value', replicate_handling='drop', baseline=3)
        pdt.assert_series_equal(obs, exp)

    def test_first_differences_baseline_invalid_baseline(self):
        with self.assertRaisesRegex(ValueError, "must be a valid state"):
            first_differences(
                metadata=qiime2.Metadata(md_one_subject_many_times),
                state_column='Time', individual_id_column='ind',
                metric='Value', replicate_handling='drop', baseline=27)

    def test_first_differences_one_subject_many_times(self):
        exp = pd.Series(
            [-0.01, 0.01, 0.01, 0.06, -0.01, 0.03, -0.02, 0.03, 0.05, -0.03,
             0.04],
            index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
            name='Difference')
        exp.index.name = '#SampleID'
        obs = first_differences(
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            metric='Value', replicate_handling='drop')
        pdt.assert_series_equal(obs, exp)

    def test_first_distances(self):
        exp = pd.Series([0.1, 0.1, 0.3, 0.1, 0.2, 0.4],
                        index=['3', '4', '5', '9', '10', '11'],
                        name='Distance')
        exp.index.name = '#SampleID'
        obs = first_distances(
            distance_matrix=dm, metadata=qiime2.Metadata(md),
            state_column='Time', individual_id_column='ind',
            replicate_handling='drop')
        pdt.assert_series_equal(obs, exp)

    def test_first_distances_single_sample(self):
        with self.assertRaisesRegex(RuntimeError, "Output is empty"):
            first_distances(
                distance_matrix=dm_single_sample, metadata=qiime2.Metadata(md),
                state_column='Time', individual_id_column='ind',
                replicate_handling='drop')

    def test_first_distances_baseline(self):
        exp = pd.Series(
            [0.3, 1.0, 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
            index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
            name='Distance')
        exp.index.name = '#SampleID'
        obs = first_distances(
            distance_matrix=dm,
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            replicate_handling='drop', baseline=0)
        pdt.assert_series_equal(obs, exp)

    def test_first_distances_baseline_is_not_state_0(self):
        exp = pd.Series(
            [0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.5, 0.8, 0.1, 0.2, 0.3],
            index=['0', '1', '2', '3', '4', '5', '6', '8', '9', '10', '11'],
            name='Distance')
        exp.index.name = '#SampleID'
        obs = first_distances(
            distance_matrix=dm,
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            replicate_handling='drop', baseline=7)
        pdt.assert_series_equal(obs, exp)

    def test_first_distances_baseline_invalid_baseline(self):
        with self.assertRaisesRegex(ValueError, "must be a valid state"):
            first_distances(
                distance_matrix=dm,
                metadata=qiime2.Metadata(md_one_subject_many_times),
                state_column='Time', individual_id_column='ind',
                replicate_handling='drop', baseline=27)

    def test_first_distances_one_subject_many_times(self):
        exp = pd.Series(
            [0.3, 0.9, 0.3, 0.2, 0.4, 0.4, 0.5, 0.8, 0.3, 0.4, 0.6],
            index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
            name='Distance')
        exp.index.name = '#SampleID'
        obs = first_distances(
            distance_matrix=dm,
            metadata=qiime2.Metadata(md_one_subject_many_times),
            state_column='Time', individual_id_column='ind',
            replicate_handling='drop')
        pdt.assert_series_equal(obs, exp)

    def test_first_distances_ecam(self):
        exp = pd.read_csv(self.get_data_path(
            'ecam-first-distances.tsv'), sep='\t', squeeze=True, index_col=0)
        obs = first_distances(
            distance_matrix=self.md_ecam_dm, metadata=self.md_ecam_fp,
            state_column='month', individual_id_column='studyid',
            replicate_handling='drop')
        pdt.assert_series_equal(obs, exp)

    def test_validate_metadata_is_superset_df(self):
        with self.assertRaisesRegex(ValueError, "Missing samples in metadata"):
            _validate_metadata_is_superset(md[md['Time'] == 1], md_dup)

    def test_validate_metadata_is_superset_distance_matrix(self):
        with self.assertRaisesRegex(ValueError, "Missing samples in metadata"):
            _validate_metadata_is_superset(
                md[md['Time'] == 1], _tabulate_matrix_ids(dm))

    def test_summarize_feature_stats(self):
        cheap_md = pd.DataFrame({'time': [1, 1, 2, 2, 3, 3]},
                                index=['s1', 's2', 's3', 's4', 's5', 's6'])
        feature_md = _summarize_feature_stats(tab, cheap_md)
        exp = pd.DataFrame(
            [[0., 0.1, 0.016, 0.5, 0.55, 0.126491106, 25.2982213],
             [-0.05, 0.05, 0.00666666667, 0.33333333333, 0.35, 0.0816496581,
              24.4948974],
             [-0.1, 0., 0.00666666667, 0.166666667, 0.15, 0.0816496581,
              48.9897949]],
            columns=['Cumulative Avg Decrease',
                     'Cumulative Avg Increase', 'Variance',
                     'Mean', 'Median', 'Standard Deviation', 'CV (%)'],
            index=['o1', 'o2', 'o3'])
        pdt.assert_frame_equal(feature_md, exp, check_names=False)


md = pd.DataFrame([(1, 'a', 0.11, 1), (1, 'a', 0.12, 2), (1, 'a', 0.13, 3),
                   (2, 'a', 0.19, 1), (2, 'a', 0.18, 2), (2, 'a', 0.21, 3),
                   (1, 'b', 0.14, 4), (1, 'b', 0.13, 5), (1, 'b', 0.14, 6),
                   (2, 'b', 0.26, 4), (2, 'b', 0.27, 5), (2, 'b', 0.29, 6)],
                  columns=['Time', 'Group', 'Value', 'ind'],
                  index=pd.Index(['0', '1', '2', '3', '4', '5',
                                  '6', '7', '8', '9', '10', '11'], name='id'))

md_one_subject_many_times = pd.DataFrame(
    [(5, 0.18, 1), (6, 0.21, 1), (7, 0.19, 1), (8, 0.22, 1), (9, 0.27, 1),
     (0, 0.12, 1), (1, 0.11, 1), (2, 0.12, 1), (3, 0.13, 1), (4, 0.19, 1),
     (10, 0.24, 1), (11, 0.28, 1)],
    columns=['Time', 'Value', 'ind'],
    index=pd.Index(['5', '6', '7', '8', '9', '0', '1', '2', '3',
                    '4', '10', '11'], name='id'))

md_static = pd.DataFrame(
    [(1, 'a', 0.11, 1), (1, 'a', 0.12, 2), (1, 'a', 0.13, 3),
     (2, 'a', 0.11, 1), (2, 'a', 0.12, 2), (2, 'a', 0.13, 3),
     (1, 'b', 0.14, 4), (1, 'b', 0.13, 5), (1, 'b', 0.14, 6),
     (2, 'b', 0.14, 4), (2, 'b', 0.13, 5), (2, 'b', 0.14, 6)],
    columns=['Time', 'Group', 'Value', 'ind'],
    index=pd.Index(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    '10', '11'], name='id'))

md_dup = pd.DataFrame([(1, 'a', 0.11, 1), (1, 'a', 0.12, 2), (1, 'a', 0.13, 2),
                       (2, 'a', 0.19, 1), (2, 'a', 0.18, 2), (2, 'a', 0.21, 3),
                       (1, 'b', 0.14, 4), (1, 'b', 0.13, 5), (1, 'b', 0.14, 6),
                       (2, 'b', 0.26, 4), (2, 'b', 0.27, 5), (2, 'b', 0.29, 6)
                       ],
                      columns=['Time', 'Group', 'Value', 'ind'],
                      index=pd.Index(['0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', '10', '11'], name='id'))

dm = skbio.DistanceMatrix.read(StringIO(
    "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\n"
    "0\t0.0\t0.3\t1.0\t0.1\t0.1\t0.3\t0.4\t0.5\t0.6\t0.1\t0.2\t0.3\n"
    "1\t0.3\t0.0\t0.9\t0.2\t0.1\t0.4\t0.2\t0.6\t0.5\t0.2\t0.3\t0.4\n"
    "2\t1.0\t0.9\t0.0\t0.3\t0.1\t0.3\t0.3\t0.6\t0.6\t0.3\t0.3\t0.4\n"
    "3\t0.1\t0.2\t0.3\t0.0\t0.2\t0.3\t0.2\t0.5\t0.4\t0.4\t0.2\t0.3\n"
    "4\t0.1\t0.1\t0.1\t0.2\t0.0\t0.4\t0.3\t0.4\t0.7\t0.1\t0.5\t0.3\n"
    "5\t0.3\t0.4\t0.3\t0.3\t0.4\t0.0\t0.4\t0.3\t0.6\t0.2\t0.4\t0.2\n"
    "6\t0.4\t0.2\t0.3\t0.2\t0.3\t0.4\t0.0\t0.5\t0.9\t0.1\t0.3\t0.1\n"
    "7\t0.5\t0.6\t0.6\t0.5\t0.4\t0.3\t0.5\t0.0\t0.8\t0.1\t0.2\t0.3\n"
    "8\t0.6\t0.5\t0.6\t0.4\t0.7\t0.6\t0.9\t0.8\t0.0\t0.3\t0.5\t0.4\n"
    "9\t0.1\t0.2\t0.3\t0.4\t0.1\t0.2\t0.1\t0.1\t0.3\t0.0\t0.4\t0.5\n"
    "10\t0.2\t0.3\t0.3\t0.2\t0.5\t0.4\t0.3\t0.2\t0.5\t0.4\t0.0\t0.6\n"
    "11\t0.3\t0.4\t0.4\t0.3\t0.3\t0.2\t0.1\t0.3\t0.4\t0.5\t0.6\t0.0\n"
    ))

dm_single_sample = skbio.DistanceMatrix.read(StringIO(
    "\t0\n"
    "0\t0.0\n"
    ))

groups = {'a': [1, 2, 3, 2, 3, 1.5, 2.5, 2.7, 3, 2, 1, 1.5],
          'b': [3, 4, 5, 4.3, 3.4, 3.2, 3, 4.3, 4.9, 5, 3.2, 3.6]}

exp_vol = pd.DataFrame(
    [(12, 7.729282, 0.005433, 0.027166),
     (6, 0.163122, .686298, 0.726866),
     (6, 0.122009, 0.726866, 0.726866),
     (6, 0.635881, 0.425206, 0.708677),
     (6, 0.996229, 0.318225, 0.708677)],
    columns=['N', 'fligner test statistic', 'P-Value', 'FDR P-value'],
    index=['All states: compare groups', 'State 1: compare groups',
           'State 2: compare groups', 'a: 1 vs. 2', 'b: 1 vs. 2'])
exp_vol.index.name = 'Comparison'

tab = pd.DataFrame({'o1': [0.3, 0.6, 0.6, 0.4, 0.5, 0.6],
                    'o2': [0.4, 0.3, 0.2, 0.4, 0.4, 0.3],
                    'o3': [0.3, 0.1, 0.2, 0.2, 0.1, 0.1]},
                   index=['s1', 's2', 's3', 's4', 's5', 's6'])

exp_tc = pd.DataFrame({(1, 'o1'): [1., 0., -1.], (1, 'o2'): [0., 1., 0.],
                       (1, 'o3'): [-1., 0., 1.], (2, 'o1'): [1., -1., 0.],
                       (2, 'o2'): [-1., 1., 0.], (2, 'o3'): [0., 0., 1.],
                       (3, 'o1'): [1., 0., 0.], (3, 'o2'): [0., 1., -1.],
                       (3, 'o3'): [0., -1., 1.]}, index=['o1', 'o2', 'o3']).T

exp_td = np.array([[0., 2., 2.], [2., 0., 2.], [2., 2., 0.]])


if __name__ == '__main__':
    unittest.main()
