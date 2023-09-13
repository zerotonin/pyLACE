"""
Survival Analysis on Fish Behavioural Responses

This script performs survival analysis on a dataset containing information on fish behaviour.
Specifically, it utilizes the Kaplan-Meier estimator and Cox Proportional-Hazards model to examine
the survival curves and hazard rates for different genotypes and sexes of fish. 
Note: The analyses do not yield statistically significant results, which could be attributed 
either to the lack of repetitions or to the specific variables like sex and genotype under consideration.

Author: Bart Geurten
"""

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import numpy as np

# Load the dataset
df = pd.read_csv('/home/bgeurten/fishDataBase/c-start_trial.csv')

# Data cleaning: Strip leading and trailing spaces from column names and column data
df.columns = df.columns.str.strip()
df['Sex'] = df['Sex'].str.strip()

# Convert 'Sex' and 'Genotype' columns to categorical data types
df['Genotype'] = df['Genotype'].astype('category')
df['Sex'] = df['Sex'].astype('category')

# Confirm that the columns are now correctly named
print(df.columns)

# Initialize the Kaplan-Meier Fitter
kmf = KaplanMeierFitter()

# Plot survival curves for Male and Female across different genotypes
for sex in ['M', 'F']:
    for label, grouped_df in df[df['Sex'] == sex].groupby('Genotype'):
        kmf.fit(grouped_df['Trial'], event_observed=grouped_df['Behav_Resp'], label=f"{label}_{sex}")
        kmf.plot()

# Separate the data by Genotype
df_Int = df[df['Genotype'] == 'Int']
df_Ht = df[df['Genotype'] == 'Ht']
df_Hm = df[df['Genotype'] == 'Hm']

# Perform log-rank tests between genotypes
results = logrank_test(df_Int['Trial'], df_Ht['Trial'], event_observed_A=df_Int['Behav_Resp'], event_observed_B=df_Ht['Behav_Resp'])
print('\nInternal vs Heterozygous')
results.print_summary()

results = logrank_test(df_Int['Trial'], df_Hm['Trial'], event_observed_A=df_Int['Behav_Resp'], event_observed_B=df_Hm['Behav_Resp'])
print('\nInternal vs Homozygous')
results.print_summary()

results = logrank_test(df_Hm['Trial'], df_Ht['Trial'], event_observed_A=df_Hm['Behav_Resp'], event_observed_B=df_Ht['Behav_Resp'])
print('\nHomozygous vs Heterozygous')
results.print_summary()

# Initialize the CoxPHFitter
cph = CoxPHFitter()

# Map the genotypes to integers for CoxPH fitting
genotype_mapping = {'Int': 0, 'Ht': 1, 'Hm': 2}
df['Genotype'] = df['Genotype'].map(genotype_mapping)
df['Genotype'] = df['Genotype'].apply(int)

# Fit the Cox Proportional-Hazards model
try:
    cph.fit(df, duration_col='Trial', event_col='Behav_Resp')
    cph.print_summary()
except Exception as e:
    print("Error occurred:", e)
