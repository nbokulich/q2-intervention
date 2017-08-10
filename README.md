# q2-intervention

[![Build Status](https://travis-ci.org/qiime2/q2-intervention.svg?branch=master)](https://travis-ci.org/qiime2/q2-intervention) [![Coverage Status](https://coveralls.io/repos/github/qiime2/q2-intervention/badge.svg?branch=master)](https://coveralls.io/github/qiime2/q2-intervention?branch=master)

QIIME2 plugin for paired sample comparisons.

q2-intervention's actions support statistical and visual comparisons of paired samples, to determine if/how samples change between observational "states". "States" will most commonly be related to time, and the sample pairs should typically consist of the same individual subject  observed at two different time points. For example, patients in a clinical study whose stool samples are collected before and after receiving treatment.

"States" can also commonly be methodological, in which case sample pairs will usually be the same individual at the same time with two different methods. For example, q2-feature-classifier could compare the effects of different collection methods, storage methods, DNA extraction methods, or any bioinformatic processing steps on the feature composition of individual samples.

In the examples below, we use data from a longitudinal study of infants' and mothers' microbiota from birth through 2 years of life ([doi: 10.1126/scitranslmed.aad7121](http://stm.sciencemag.org/content/8/343/343ra82)). These data are located in the directory [`q2_intervention/tutorial_data/`](https://github.com/qiime2/q2-intervention/tree/master/q2_intervention/tutorial_data).

## Examples

### Paired difference testing

Paired difference tests determine whether the value of a specific metric changed significantly between pairs of paired samples (e.g., pre- and post-treatment).

This visualizer currently supports comparison of feature abundance (e.g., microbial sequence variants or taxa) in a feature table, or of metadata values in a sample metadata file. Alpha diversity values (e.g., observed sequence variants) and beta diversity values (e.g., principal coordinates) are useful metrics for comparison with these tests, and should be contained in one of the metadata files given as input.

#### Paired differences in metadata

Here we use `pairwise-differences` to assess whether alpha diversity (here, Shannon's diversity index) changed significantly between 0 and 12 months of life in vaginally born and Cesarean-delivered infants, and whether the magnitude of change differed between these groups. Note that the alpha diversity data in this case is contained in a separate artifact, which is the typical (and preferred) approach; alternatively, alpha diversity data (or other data) contained in a sample metadata file can be used as the input `metric`, in which case the second `--m-metadata-file` input is unnecessary.

```
qiime intervention pairwise-differences \
    --m-metadata-file ecam_map_maturity.txt \
    --m-metadata-file ecam_shannon.qza \
    --p-metric shannon \
    --p-group-column delivery \
    --p-state-column month \
    --p-state-1 0 \
    --p-state-2 12 \
    --p-individual-id-column studyid \
    --o-visualization ecam-delivery-alpha \
    --p-no-drop-duplicates
```

#### Pairwise differences in feature table

We can also use this method to measure changes in the abundances of specific features of interest. In this example, we test whether the abundance of genus Bacteroides changed significantly between 6 and 18 months of life in vaginally born and Cesarean-delivered infants, and whether the magnitude of change differed between these groups. Note that `pairwise-differences` accepts a feature table as optional input to extract taxon abundance data.

```
qiime intervention pairwise-differences \
    --i-table ecam-table-taxa.qza \
    --m-metadata-file ecam_map_maturity.txt \
    --p-metric 'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__' \
    --p-group-column delivery \
    --p-state-column month \
    --p-state-1 6 \
    --p-state-2 18 \
    --p-individual-id-column studyid \
    --o-visualization ecam-delivery \
    --p-no-drop-duplicates
```

### Pairwise distance testing

The `pairwise-distances` visualizer also assesses changes between paired samples from two different "states", but instead of taking a metadata column or feature ID as input, it operates on a distance matrix to assess the distance between "pre" and "post" sample pairs. The "within-subject" distance between paired samples from an individual are always calculated for each group in the metadata `group_column`; by default, "between-subject" distances between all individuals in a given `group_column` are also calculated and compared. Between-subject distances include all samples sharing the same `group_column` that are not pairs of "within-subject" samples from `state_1` and `state_2`, but otherwise ignore the `state_column` and `individual_id_column` parameters, so will pair all samples from all time points (or whatever the comparison "state" is) in the distance matrix. Hence, users should carefully consider what type of comparison they wish to perform and, if appropriate, filter the distance matrix prior to using this visualizer. Filtering can be performed with `filter-distance-matrix` as described [here](https://docs.qiime2.org/2017.5/tutorials/filtering/#filtering-distance-matrices).

In this example, we test whether an individual's stool microbiota (as assessed by unweighted UniFrac distance) differs significantly between 0 and 12 months of life in vaginally born and Cesarean-delivered infants, and whether the within- and between-subject distances differed between these groups. 
```
qiime intervention pairwise-distances \
    --i-distance-matrix ecam-unweighted-distance-matrix.qza \
    --m-metadata-file ecam_map_maturity.txt \
    --p-group-column delivery \
    --p-state-column month \
    --p-state-1 0 \
    --p-state-2 12 \
    --p-individual-id-column studyid \
    --o-visualization ecam-delivery-distance \
    --p-no-drop-duplicates
```

If between-subject distances are not important, the same visualization can be performed excluding these distances with the following command:
```
qiime intervention pairwise-distances \
    --i-distance-matrix ecam-unweighted-distance-matrix.qza \
    --m-metadata-file ecam_map_maturity.txt \
    --p-group-column delivery \
    --p-state-column month \
    --p-state-1 0 \
    --p-state-2 12 \
    --p-individual-id-column studyid \
    --o-visualization ecam-delivery-distance-no-between \
    --p-no-drop-duplicates \
    --p-no-between-group-distance
```

### Linear mixed effects models

Linear mixed effects models test the relationship between a single response variable and one or more independent variables. This implementation takes at least one numeric "state_column" (e.g., Time) and one or more comma-separated group_categories (which may be categorical or numeric) as independent variables in a LME model, and plots regression plots of the response variable ("metric") as a function of the state caregory and each group column. The response variable may either be a sample metadata mapping file column or a feature ID in the feature table.

In this example, we demonstrate the use of `linear-mixed-effects` to test the relationship between `shannon` (alpha diversity), age, delivery mode, diet, and sex.

```
qiime intervention linear-mixed-effects \
    --m-metadata-file ecam_map_maturity.txt \
    --p-metric shannon \
    --m-metadata-file ecam_shannon.qza \
    --p-group-categories delivery,diet,sex \
    --p-state-column month \
    --p-individual-id-column studyid \
    --o-visualization ecam-lme
```


Second, we demonstrate the use of `linear-mixed-effects` to test the relationship between `Bacteroides`, age, delivery mode, diet, and sex.

```
qiime intervention linear-mixed-effects \
    --i-table ecam-table-taxa.qza \
    --m-metadata-file ecam_map_maturity.txt \
    --p-metric 'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__' \
    --p-group-categories delivery,diet,sex \
    --p-state-column month \
    --p-individual-id-column studyid \
    --o-visualization ecam-bacteroides-lme
```
