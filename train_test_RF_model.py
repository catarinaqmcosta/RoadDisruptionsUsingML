import pandas as pd
import numpy as np
import os
import time
import random
import utils
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

out_dir = './outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

data_train = pd.read_csv('./inputs/training_dataset_Lx.csv')
data_test = pd.read_csv('./inputs/testing_dataset_Lx.csv')

grid_proba_out = 'Grid_probabilities_train_vs_test.csv'
sim_code = 'train_vs_test'


# Training and testing datasets
X_train_ = data_train[IVS]
X_test_ = data_test[IVS]
y_train = data_train[DV]
y_test = data_test[DV]


################################################
# Random Forest Classifier
################################################

# Exclude the Grid fid column during classification
X_train = X_train_.drop(columns=['Grid fid'])
X_test = X_test_.drop(columns=['Grid fid'])

# rf = RandomForestClassifier(max_depth=14,
                            # max_features='log2',
                            # n_estimators=106)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

predicted_proba = rf.predict_proba(X_test)
y_pred = (predicted_proba[:, 1] >= 0.5).astype('int')

################################################
# Calculate probabilities of disruption in each grid cell
################################################
# Calculate the average of all test values for each grid cell
test_df = X_test_[['Grid fid']]
test_df = test_df.copy()
test_df.loc[:, 'Grid_prob_test'] = y_test
test_df.loc[:, 'Grid_prob_test_pred'] = y_pred
test_grid = test_df.groupby(by=["Grid fid"]).mean()

# Calculate the average of all train values for each grid cell
train_df = X_train_[['Grid fid']]
train_df = train_df.copy()
train_df.loc[:, 'Grid_prob_train'] = y_train
train_grid = train_df.groupby(by=["Grid fid"]).mean()

# Store train and test results per grid cell
grid_prob = data_train[['Grid fid']].drop_duplicates()
grid_prob = pd.merge(grid_prob, test_grid, on=['Grid fid'])
grid_prob = pd.merge(grid_prob, train_grid, on=['Grid fid'])
grid_prob['abs_diff_train_vs_test'] = abs(grid_prob['Grid_prob_test'] -
                                          grid_prob['Grid_prob_train'])
grid_prob['abs_diff_test_vs_pred'] = abs(grid_prob['Grid_prob_test'] -
                                         grid_prob['Grid_prob_test_pred'])

# Save grid probabilities to csv
grid_prob.to_csv(os.path.join(out_dir, grid_proba_out), index=0)


################################################
# Evaluation metrics
################################################
utils.save_class_report(y_test, y_pred, sim_code, out_dir)

utils.plot_ROC_curve_diff_thre(y_test, y_pred, predicted_proba, sim_code,
                               out_dir)

utils.PR_curves_diff_thre(y_test, y_pred, predicted_proba, sim_code, out_dir)

utils.confusion_matrix_diagonal(grid_prob['Grid_prob_test_pred'],
                                grid_prob['Grid_prob_test'],
                                "Disruption probability from the test dataset",
                                "Disruption probability predicted by the model",
                                "Grid probabilities", sim_code, out_dir)

utils.plot_conf_matrix(y_test, y_pred, sim_code, out_dir)


print("--- %s seconds ---" % (time.time() - start_time))
