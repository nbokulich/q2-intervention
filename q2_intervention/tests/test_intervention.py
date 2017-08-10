# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2
import pandas as pd
from skbio import DistanceMatrix
from io import StringIO
from warnings import filterwarnings
from q2_intervention._utilities import (
    _get_group_pairs, _extract_distance_distribution,
    _get_pairwise_differences,
    _between_subject_distance_distribution, _compare_pairwise_differences,
    _multiple_group_difference, _per_method_pairwise_stats)
from q2_intervention._intervention import (
    pairwise_differences, pairwise_distances, linear_mixed_effects)
import tempfile
import pkg_resources
from qiime2.plugin.testing import TestPluginBase


filterwarnings("ignore", category=UserWarning)


class InterventionTestPluginBase(TestPluginBase):
    package = 'q2_intervention.tests'

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='q2-intervention-test-temp-')

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_data_path(self, filename):
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)


class UtilitiesTests(InterventionTestPluginBase):

    def test_get_group_pairs(self):
        res = _get_group_pairs(
            md, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2])
        self.assertEqual(res, [('0', '3'), ('1', '4'), ('2', '5')])
        res = _get_group_pairs(
            md_dup, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2])
        self.assertEqual(res, [('0', '3')])
        res = _get_group_pairs(
            md_dup, 'a', individual_id_column='ind', group_column='Group',
            state_column='Time', state_values=[1, 2], drop_duplicates=False)
        self.assertEqual(res[0], ('0', '3'))
        self.assertIn(res[1], [('1', '4'), ('2', '4')])

    def test_extract_distance_distribution(self):
        res = _extract_distance_distribution(dm, [('0', '3'), ('2', '5')])
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
        res = _get_pairwise_differences(
            md, [('0', '3'), ('1', '4'), ('2', '5')], 'Value')
        self.assertEqual(res, [0.08, 0.06, 0.07999999999999999])

    def test_compare_pairwise_differences_parametric(self):
        res = _compare_pairwise_differences(groups, parametric=True)
        self.assertAlmostEqual(res['FDR P']['a'], 9.4882148564067405e-07)
        self.assertAlmostEqual(res['FDR P']['b'], 4.8474685173462082e-09)

    def test_compare_pairwise_differences_nonparametric(self):
        res = _compare_pairwise_differences(groups, parametric=False)
        self.assertAlmostEqual(res['FDR P']['a'], 0.0021830447373622506)
        self.assertAlmostEqual(res['FDR P']['b'], 0.0021830447373622506)

    def test_multiple_group_difference_parametric(self):
        res = _multiple_group_difference(groups.values(), parametric=True)
        self.assertAlmostEqual(res['P value'], 7.6936106994369541e-06)

    def test_multiple_group_difference_nonparametric(self):
        res = _multiple_group_difference(groups.values(), parametric=False)
        self.assertAlmostEqual(res['P value'], 6.0957929139040073e-05)

    def test_per_method_pairwise_stats_unpaired_parametric(self):
        res = _per_method_pairwise_stats(groups, paired=False, parametric=True)
        self.assertAlmostEqual(res['FDR P'][0], 7.693610699436966e-06)

    def test_per_method_pairwise_stats_unpaired_nonparametric(self):
        res = _per_method_pairwise_stats(
            groups, paired=False, parametric=False)
        self.assertAlmostEqual(res['FDR P'][0], 6.890936276106502e-05)

    def test_per_method_pairwise_stats_paired_parametric(self):
        res = _per_method_pairwise_stats(groups, paired=True, parametric=True)
        self.assertAlmostEqual(res['FDR P'][0], 3.085284368834677e-06)

    def test_per_method_pairwise_stats_paired_nonparametric(self):
        res = _per_method_pairwise_stats(groups, paired=True, parametric=False)
        self.assertAlmostEqual(res['FDR P'][0], 0.0021830447373622506)


# This test class really just makes sure that each plugin runs without error.
# UtilitiesTests handles all stats under the hood, so here we just want to make
# sure all plugins run smoothly.
class InterventionTests(InterventionTestPluginBase):

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
        self.md_ecam_fp = _load_md('ecam_map_maturity.txt')
        self.md_ecam_dm = _load_dm('ecam-unweighted-distance-matrix.qza')

    def test_pairwise_differences(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid', metric='observed_otus')

    def test_pairwise_differences_taxa(self):
        pairwise_differences(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa')

    def test_pairwise_distances(self):
        pairwise_distances(
            output_dir=self.temp_dir.name, distance_matrix=self.md_ecam_dm,
            metadata=self.md_ecam_fp, group_column='delivery',
            state_column='month', state_1=0, state_2=3,
            individual_id_column='studyid')

    def test_linear_mixed_effects(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=None,
            metadata=self.md_ecam_fp, state_column='month',
            group_categories='delivery,diet,antiexposedall',
            individual_id_column='studyid', metric='observed_otus')

    def test_linear_mixed_effects_taxa(self):
        linear_mixed_effects(
            output_dir=self.temp_dir.name, table=self.table_ecam_fp,
            metadata=self.md_ecam_fp, state_column='month',
            group_categories='delivery,diet,antiexposedall',
            individual_id_column='studyid',
            metric='e2c3ff4f647112723741aa72087f1bfa')


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
