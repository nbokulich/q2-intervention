# ----------------------------------------------------------------------------
# Copyright (c) 2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from io import StringIO
from warnings import filterwarnings
from q2_longitudinal._utilities import (
    _get_group_pairs, _extract_distance_distribution,
    _get_pairwise_differences, _validate_input_values, _validate_input_columns,
    _between_subject_distance_distribution, _compare_pairwise_differences,
    _multiple_group_difference, _per_method_pairwise_stats,
    _calculate_variability, _multiple_tests_correction,
    _add_sample_size_to_xtick_labels, _temporal_corr, _temporal_distance,
    _nmit, _massage_data, _drop_undersampled_data)
from q2_longitudinal._longitudinal import (
    pairwise_differences, pairwise_distances, linear_mixed_effects, volatility,
    nmit)
import tempfile
import pkg_resources
from qiime2.plugin.testing import TestPluginBase


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)


class longitudinalTestPluginBase(TestPluginBase):
    package = 'q2_longitudinal.tests'

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='q2-longitudinal-test-temp-')

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_data_path(self, filename):
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)


class UtilitiesTests(longitudinalTestPluginBase):

    def test_get_group_pairs(self):
        res, err = _get_group_pairs(
            md, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2],
            replicate_handling='drop')
        self.assertEqual(res, [('0', '3'), ('1', '4'), ('2', '5')])
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

    def test_calculate_variability(self):
        res = _calculate_variability(md, 'Value')
        for obs, exp in zip(res, [0.1808333, 0.0634548, 0.3711978, -0.0095311,
                                  0.30774299, 0.05392367]):
            self.assertAlmostEqual(obs, exp)

    def test_add_sample_size_to_xtick_labels(self):
        groups = {'a': [1, 2, 3], 'b': [1, 2], 'c': [1, 2, 3]}
        labels = _add_sample_size_to_xtick_labels(groups)
        self.assertEqual(labels, ['a (n=3)', 'b (n=2)', 'c (n=3)'])

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
        self.assertTrue(test_df_mt.sort_index().equals(test_df.sort_index()))

    def test_temporal_corr(self):
        ind_id = pd.Series(
            [1, 2, 3, 1, 2, 3], index=['s1', 's2', 's3', 's4', 's5', 's6'])
        obs_tc = _temporal_corr(tab, ind_id)
        for k in obs_tc.keys():
            self.assertTrue(
                exp_tc[k].sort_index().equals(obs_tc[k].sort_index()))

    def test_temporal_distance(self):
        id_set = pd.Series([1, 2, 3], index=['s1', 's2', 's3'])
        obs_td = _temporal_distance(exp_tc, id_set)
        self.assertTrue(np.array_equal(obs_td.data, exp_td))

    def test_nmit(self):
        sample_md = pd.DataFrame([1, 2, 3, 1, 2, 3], columns=['sample_id'],
                                 index=['s1', 's2', 's3', 's4', 's5', 's6'])
        obs_td = _nmit(tab, sample_md, 'sample_id')
        self.assertTrue(np.array_equal(obs_td.data, exp_td))

    def test_drop_undersampled_data_drop_by_state(self):
        exp = pd.DataFrame({'time': [7, 7, 7, 7, 7],
                            'id': [1, 2, 3, 3, 4]},
                           index=[14, 15, 16, 17, 18])
        obs, error = _drop_undersampled_data(
            df_test_drop_undersampled_data, 'time', auto=False)
        self.assertTrue(obs.sort_index().equals(exp.sort_index()))

    def test_drop_undersampled_data_drop_by_subject(self):
        exp = pd.DataFrame({'time': [1, 2, 3, 5, 6, 7, 7, 8],
                            'id': [3, 3, 3, 3, 3, 3, 3, 3]},
                           index=[2, 5, 8, 12, 13, 16, 17, 19])
        obs, error = _drop_undersampled_data(
            df_test_drop_undersampled_data, 'id', auto=False)
        self.assertTrue(obs.sort_index().equals(exp.sort_index()))

    def test_drop_undersampled_data_auto(self):
        exp = pd.DataFrame(
            {'time': [1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 7, 7, 7, 7],
             'id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 4]},
            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18])
        obs, error = _drop_undersampled_data(
            df_test_drop_undersampled_data, 'time', auto=True)
        self.assertTrue(obs.sort_index().equals(exp.sort_index()))

    def test_massage_data_auto(self):
        exp = pd.DataFrame({'time': [1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 7, 7, 7],
                            'id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3]},
                           index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17])
        obs, error = _massage_data(df_test_drop_undersampled_data, 'time',
                                   'id', 'ignore', 'auto')
        self.assertTrue(obs.sort_index().equals(exp.sort_index()))

    def test_massage_data_duplicates(self):
        exp = pd.DataFrame({'time': [1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 7, 7],
                            'id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]},
                            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16])
        obs, error = _massage_data(
            df_test_drop_undersampled_data, 'time', 'id', 'random', 'auto')
        self.assertTrue(obs['time'].equals(exp['time']))
        self.assertTrue(obs['id'].equals(exp['id']))

    def test_massage_data_bad_param_combo_error(self):
        with self.assertRaisesRegex(ValueError, "Do not set"):
            obs, error = _massage_data(md_dup, 'Time', 'ind', 'drop', 'auto')

    def test_massage_data_all_samples_filtered(self):
        with self.assertRaisesRegex(RuntimeError, "filtered out!"):
            obs, error = _massage_data(df_test_drop_undersampled_data, 'time',
                                       'id', 'random', 'individuals')


# This test class really just makes sure that each plugin runs without error.
# UtilitiesTests handles all stats under the hood, so here we just want to make
# sure all plugins run smoothly.
class longitudinalTests(longitudinalTestPluginBase):

    def setUp(self):
        super().setUp()

        def _load_features(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(pd.DataFrame)
            return table

        def _load_md(md_fp):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.Metadata(md)
            return md

        def _load_dm(dm_fp):
            dm_fp = self.get_data_path(dm_fp)
            dm = qiime2.Artifact.load(dm_fp)
            dm = dm.view(DistanceMatrix)
            return dm

        self.table_ecam_fp = _load_features('ecam-table-maturity.qza')
        self.table_taxa_fp = _load_features('ecam-table-small.qza')
        self.md_ecam_fp = _load_md('ecam_map_maturity.txt')
        self.md_ecam_dm = _load_dm('ecam-unweighted-distance-matrix.qza')

    def test_validate_input_values(self):
        with self.assertRaisesRegex(ValueError, "state_1 and state_2"):
            _validate_input_values(md, "ind", "Group", "Time", 1, 1)
        with self.assertRaisesRegex(ValueError, "not present"):
            _validate_input_values(md, "ind", "Group", "Time", 1, 3)
        with self.assertRaisesRegex(ValueError, "not a column"):
            _validate_input_values(md, "ind", "Group", "Days", 1, 2)
        with self.assertRaisesRegex(ValueError, "not a column"):
            _validate_input_columns(md, "ind", ["Group", "More stuff"], "Time")
        dropped = md.drop(['9', '10', '11'])
        with self.assertRaisesRegex(ValueError, "not represented"):
            _validate_input_values(dropped, "ind", "Group", "Time", 1, 2)
        with self.assertRaisesRegex(ValueError, "state_1 and state_2"):
            pairwise_differences(
                output_dir=self.temp_dir.name, table=None,
                metadata=self.md_ecam_fp, group_column='delivery',
                state_column='month', state_1=0, state_2=0,
                individual_id_column='studyid', metric='observed_otus',
                replicate_handling='drop')

    def test_pairwise_differences(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, group_column='delivery',
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
            group_categories='delivery,diet,antiexposedall',
            individual_id_column='studyid', metric='observed_otus')

    def test_linear_mixed_effects_one_variable(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_categories='delivery',
            individual_id_column='studyid', metric='observed_otus')

    def test_linear_mixed_effects_taxa(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, state_column='month',
            group_categories='delivery,diet,antiexposedall',
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa')

    def test_volatility(self):
        volatility(
            output_dir=self.temp_dir.name, metadata=self.md_ecam_fp,
            metric='observed_otus', group_column='delivery',
            state_column='month', individual_id_column='studyid')

    def test_linear_mixed_effects_singular_matrix_error(self):
        with self.assertRaisesRegex(ValueError, "singular matrix error"):
            linear_mixed_effects(
                output_dir=self.temp_dir.name, table=None,
                metadata=self.md_ecam_fp, state_column='month',
                group_categories='diet,diet_3',
                individual_id_column='studyid', metric='observed_otus')

    def test_nmit(self):
        nmit(table=self.table_taxa_fp, metadata=self.md_ecam_fp,
             individual_id_column='studyid')

    def test_nmit_missing_table_ids(self):
        md = qiime2.Metadata(pd.DataFrame([[1]], columns=['i'], index=['20']))
        with self.assertRaisesRegex(ValueError, 'Missing samples'):
            nmit(table=self.table_taxa_fp, metadata=md,
                 individual_id_column='studyid')


md = pd.DataFrame([(1, 'a', 0.11, 1), (1, 'a', 0.12, 2), (1, 'a', 0.13, 3),
                   (2, 'a', 0.19, 1), (2, 'a', 0.18, 2), (2, 'a', 0.21, 3),
                   (1, 'b', 0.14, 4), (1, 'b', 0.13, 5), (1, 'b', 0.14, 6),
                   (2, 'b', 0.26, 4), (2, 'b', 0.27, 5), (2, 'b', 0.29, 6)],
                  columns=['Time', 'Group', 'Value', 'ind'],
                  index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         '10', '11'])

md_dup = pd.DataFrame([(1, 'a', 0.11, 1), (1, 'a', 0.12, 2), (1, 'a', 0.13, 2),
                       (2, 'a', 0.19, 1), (2, 'a', 0.18, 2), (2, 'a', 0.21, 3),
                       (1, 'b', 0.14, 4), (1, 'b', 0.13, 5), (1, 'b', 0.14, 6),
                       (2, 'b', 0.26, 4), (2, 'b', 0.27, 5), (2, 'b', 0.29, 6)
                       ],
                      columns=['Time', 'Group', 'Value', 'ind'],
                      index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11'])

dm = DistanceMatrix.read(StringIO(
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

groups = {'a': [1, 2, 3, 2, 3, 1.5, 2.5, 2.7, 3, 2, 1, 1.5],
          'b': [3, 4, 5, 4.3, 3.4, 3.2, 3, 4.3, 4.9, 5, 3.2, 3.6]}

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

df_test_drop_undersampled_data = pd.DataFrame(
    {'time': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
              5, 5, 6, 7, 7, 7, 7, 7, 8, 8, 9],
     'id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 3, 1, 2, 3, 3, 4, 3, 4, 9]})
