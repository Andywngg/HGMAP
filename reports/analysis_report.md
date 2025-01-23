# Microbiome Analysis Report
Generated on: 2025-01-23 13:53:10

## 1. Dataset Overview
Total samples analyzed: 8070
Total features: 65

### 1.1 Health Status Distribution
```
Subject health status (Healthy or Non-healthy)
Healthy        5547
Non-healthy    2522
```

### 1.2 Phenotype Distribution
Top 10 phenotypes:
```
Phenotype
Healthy                                   5547
Colorectal cancer                          789
Type 2 diabetes                            377
Crohn's disease                            284
Ulcerative colitis                         241
Atherosclerotic cardiovascular disease     214
Liver Cirrhosis                            152
Rheumatoid arthritis                       102
Grave's disease                            100
Ankylosing spondylitis                      95
```

### 1.3 Geographic Distribution
Top 10 countries:
```
Geography (Country)
China             2324
USA               1573
Israel             900
United Kingdom     656
Netherlands        576
Japan              286
Denmark            285
Germany            206
Italy              190
Spain              112
```

### 1.4 Demographics
Age Statistics:
```
count    4670.000000
mean       45.633345
std        17.940496
min         0.000000
25%        30.000000
50%        46.000000
75%        60.962355
max       107.000000
```

BMI Statistics:
```
count    3524.000000
mean       22.819414
std         3.132911
min        13.219035
25%        20.908314
50%        22.640432
75%        24.164522
max        78.500000
```

## 2. Diversity Analysis
### 2.1 Diversity Metrics Summary
```
  Unnamed: 0  shannon_diversity  species_richness     evenness
0      count        8070.000000       8070.000000  8070.000000
1       mean           0.010698         78.085502     0.002453
2        std           0.071343          8.949169     0.016089
3        min           0.001693         63.000000     0.000384
4        25%           0.002252         70.000000     0.000518
5        50%           0.003282         78.000000     0.000754
6        75%           0.006206         86.000000     0.001426
7        max           3.993563         93.000000     0.889705
```

## 3. Visualizations
The following visualizations have been generated:
1. `diversity_distributions.png`: Distribution of diversity metrics
2. `pca_explained_variance.png`: Cumulative explained variance by PCA components
3. `tsne_visualization.png`: t-SNE visualization of samples colored by health status
4. `metadata_correlations.png`: Correlation heatmap between metadata and diversity metrics

## 4. Key Findings
- Average Shannon diversity: 0.01
- Average species richness: 78.09
- Healthy samples comprise 68.7% of the dataset
- Samples collected from 26 different countries