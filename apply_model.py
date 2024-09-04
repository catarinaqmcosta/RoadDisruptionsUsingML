import pandas as pd
import numpy as np
import os
import time
import random
from sklearn.ensemble import RandomForestClassifier
random.seed(100)
np.random.seed(100)


start_time = time.time()
################################################
# Input data
################################################

# Independent variables
IVS = [
        'GHS_BUILT_S (m2)',
        'GHS_POP',
        'Nr buildings mid rise',
        'Nr buildings high rise',
        'Number of buildings in the grid cell',
        'Collapsed mid rise',
        'Collapsed high rise',
        'Total collapses (%)',
        'OSM length (m)',
        'OSM area (m2)',
        'Open space (m2)',
        'Avg road width (m)',
        'Grid fid',
        'Total collapses',
        'OSM max road width (m)',
        'OSM nr nodes',
        'Collapsed low rise',
        'GHS_BUILT_V (m3)',
        'GHS_BUILT_H (m)',
        'Nr buildings low rise'
        ]
# Dependent variable
DV = ['Affected length (bin)']

out_dir = './outputs_scOnshore/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Training dataset
data_train = pd.read_csv('./inputs/training_dataset_Lx.csv')

# inputs - real scenario
sc_dataset = pd.read_csv('./inputs_Sc_onshore/Lisbon_scOnshore_disagg_GHS_V.csv')
nr_scenarios = 200

out_prob = 'Grid_prob_sc_Onshore_Lx_disagg_GHS_V.csv'


X_train_ = data_train[IVS]
y_train = data_train[DV]

sc_dataset = sc_dataset[IVS]
X_sc = sc_dataset.drop(columns=['Grid fid'])

################################################
# Random Forest Classifier
################################################

# Exclude the Grid fid column during classification
X_train = X_train_.drop(columns=['Grid fid'])

# rf = RandomForestClassifier(max_depth=14,
                            # max_features='log2',
                            # n_estimators=106)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

predicted_proba = rf.predict_proba(X_sc)
y_pred = (predicted_proba[:, 1] >= 0.5).astype('int')

################################################
# Calculate probabilities of disruption in each grid cell
################################################
test_df = sc_dataset[['Grid fid']]
test_df = test_df.copy()
test_df.loc[:, 'Grid_prob_pred'] = y_pred
test_grid = test_df.groupby(by=["Grid fid"]).sum()/nr_scenarios
print(test_grid)

# Calculate affected pop in each grid cell
pop = sc_dataset[['GHS_POP', 'Grid fid']]
pop = pop.drop_duplicates()
test_grid = pd.merge(test_grid, pop, on="Grid fid")
test_grid['pop_affected'] = test_grid['GHS_POP']*test_grid['Grid_prob_pred']

test_grid = test_grid.drop(columns=['GHS_POP'])

test_grid.to_csv(os.path.join(out_dir, out_prob), index=0)
