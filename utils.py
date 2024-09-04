# Data Processing
import pandas as pd
import numpy as np
import os

# Modelling
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve

# Tree Visualisation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from textwrap import wrap


################################################
# Confusion Matrix
################################################
def plot_conf_matrix(y_test, y_pred, sim_code, out_dir):
    # option 1 to plot confusion matrix
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.set(font_scale=2)
    x_axis_labels = ['Without road disruptions', 'With road disruptions']
    x_axis_labels = ['\n'.join(wrap(ab, 15)) for ab in x_axis_labels]
    sns.heatmap(cm, annot=labels, xticklabels=x_axis_labels,
                yticklabels=x_axis_labels,
                fmt="", cmap=sns.cubehelix_palette(as_cmap=True),
                annot_kws={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('True label', fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{sim_code}.png'))
    # plt.show()
    plt.close()
    sns.set(font_scale=1)


################################################
# ROC curve
################################################
def plot_ROC_curve_diff_thre(y_test, y_pred, predicted_proba, sim_code,
                             out_dir):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    fpr_, tpr_, proba = roc_curve(y_test, predicted_proba[:, -1])
    auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC=%.3f' % (auc))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    # for specific decision thresholds
    y_pred_01 = (predicted_proba[:, 1] >= 0.1).astype('int')
    fpr01, tpr01, _ = roc_curve(y_test, y_pred_01)
    y_pred_02 = (predicted_proba[:, 1] >= 0.2).astype('int')
    fpr02, tpr02, _ = roc_curve(y_test, y_pred_02)
    y_pred_03 = (predicted_proba[:, 1] >= 0.3).astype('int')
    fpr03, tpr03, _ = roc_curve(y_test, y_pred_03)
    y_pred_04 = (predicted_proba[:, 1] >= 0.4).astype('int')
    fpr04, tpr04, _ = roc_curve(y_test, y_pred_04)
    y_pred_06 = (predicted_proba[:, 1] >= 0.6).astype('int')
    fpr06, tpr06, _ = roc_curve(y_test, y_pred_06)
    y_pred_07 = (predicted_proba[:, 1] >= 0.7).astype('int')
    fpr07, tpr07, _ = roc_curve(y_test, y_pred_07)
    y_pred_08 = (predicted_proba[:, 1] >= 0.8).astype('int')
    fpr08, tpr08, _ = roc_curve(y_test, y_pred_08)
    y_pred_09 = (predicted_proba[:, 1] >= 0.9).astype('int')
    fpr09, tpr09, _ = roc_curve(y_test, y_pred_09)

    plt.plot(fpr_, tpr_, linewidth=0.75, c='grey')

    plt.scatter(fpr01[1], tpr01[1], marker='.', s=150, label='0.1')
    plt.scatter(fpr02[1], tpr02[1], marker='.', s=150, label='0.2')
    plt.scatter(fpr03[1], tpr03[1], marker='.', s=150, label='0.3')
    plt.scatter(fpr04[1], tpr04[1], marker='.', s=150, label='0.4')
    plt.scatter(fpr[1], tpr[1], marker='.', s=150, label='0.5')
    plt.scatter(fpr06[1], tpr06[1], marker='.', s=150, label='0.6')
    plt.scatter(fpr07[1], tpr07[1], marker='.', s=150, label='0.7')
    plt.scatter(fpr08[1], tpr08[1], marker='.', s=150, label='0.8')
    plt.scatter(fpr09[1], tpr09[1], marker='.', s=150, label='0.9')

    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, c='grey')

    plt.axis('square')
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])

    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    legend = plt.legend(frameon=False)
    legend.set_title("Threshold")
    plt.savefig(os.path.join(out_dir, f'roc_curve_{sim_code}.png'))
    # plt.show()
    plt.close()


################################################
# Precision recall curve for different decision thresholds
################################################
def PR_curves_diff_thre(y_test, y_pred, predicted_proba, sim_code, out_dir):
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    precision_, recall_, proba = precision_recall_curve(y_test,
                                                        predicted_proba[:, -1])
    plt.plot(recall_, precision_, linewidth=.75, c='grey')
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_),
                                  proba)),
                                  key=lambda i: i[0], reverse=False)[0][1]
    print(f"Optimal decision threshold = {optimal_proba_cutoff}")

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    # for specific decision thresholds
    y_pred_01 = (predicted_proba[:, 1] >= 0.1).astype('int')
    precision01, recall01, _ = precision_recall_curve(y_test, y_pred_01)
    y_pred_02 = (predicted_proba[:, 1] >= 0.2).astype('int')
    precision02, recall02, _ = precision_recall_curve(y_test, y_pred_02)
    y_pred_03 = (predicted_proba[:, 1] >= 0.3).astype('int')
    precision03, recall03, _ = precision_recall_curve(y_test, y_pred_03)
    y_pred_04 = (predicted_proba[:, 1] >= 0.4).astype('int')
    precision04, recall04, _ = precision_recall_curve(y_test, y_pred_04)
    y_pred_06 = (predicted_proba[:, 1] >= 0.6).astype('int')
    precision06, recall06, _ = precision_recall_curve(y_test, y_pred_06)
    y_pred_07 = (predicted_proba[:, 1] >= 0.7).astype('int')
    precision07, recall07, _ = precision_recall_curve(y_test, y_pred_07)
    y_pred_08 = (predicted_proba[:, 1] >= 0.8).astype('int')
    precision08, recall08, _ = precision_recall_curve(y_test, y_pred_08)
    y_pred_09 = (predicted_proba[:, 1] >= 0.9).astype('int')
    precision09, recall09, _ = precision_recall_curve(y_test, y_pred_09)

    plt.scatter(recall01[1], precision01[1], marker='.', s=150, label='0.1')
    plt.scatter(recall02[1], precision02[1], marker='.', s=150, label='0.2')
    plt.scatter(recall03[1], precision03[1], marker='.', s=150, label='0.3')
    plt.scatter(recall04[1], precision04[1], marker='.', s=150, label='0.4')
    plt.scatter(recall[1], precision[1], marker='.', s=150, label='0.5')
    plt.scatter(recall06[1], precision06[1], marker='.', s=150, label='0.6')
    plt.scatter(recall07[1], precision07[1], marker='.', s=150, label='0.7')
    plt.scatter(recall08[1], precision08[1], marker='.', s=150, label='0.8')
    plt.scatter(recall09[1], precision09[1], marker='.', s=150, label='0.9')
    plt.axis('square')
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    legend = plt.legend(frameon=False)
    legend.set_title("Threshold")
    plt.savefig(os.path.join(out_dir,
                             f'precision_recall_curve_{sim_code}.png'))
    # plt.show()
    plt.close()


def save_class_report(y_test, y_pred, sim_code, out_dir):
    report = classification_report(y_test, y_pred)
    lines = report.split('\n')
    plotMat = []
    for line in lines[2:(len(lines)-3)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        v = [float(x) for x in t[1: len(t)]]
        plotMat.append(v)

    out = pd.DataFrame(columns=['precision', 'recall', 'f1score', 'support'],
                       data=plotMat)
    out.to_csv(os.path.join(out_dir, f'class_report_{sim_code}.csv'), index=0)


################################################
# Plot confusion matrix of grid probabilities
################################################

def confusion_matrix_diagonal(pred_col, real_col, xlabel, ylabel, title,
                              sim_code, out_dir):
    pred_col = pred_col.round(decimals=1)
    real_col = real_col.round(decimals=1)

    unique_values_x = np.sort(real_col.unique())
    unique_values_y = np.sort(pred_col.unique())

    matrix = pd.crosstab(pred_col, real_col)
    print(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        if matrix.shape[0]-matrix.shape[1] == -1:
            matrix.loc[unique_values_x[-1]] = int(0)
        elif matrix.shape[0]-matrix.shape[1] == 1:
            matrix.loc[:, unique_values_y[-1]] = int(0)
        else:
            print("TODO: add support for other non-square cases")
            exit()
    print(matrix.index.values)
    print(matrix.columns.values)

    # For rectangular matrix:
    # xticklabels = unique_values_x
    # yticklabels = unique_values_y
    # For square matrix:
    xticklabels = yticklabels = matrix.index.values

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks(np.arange(len(xticklabels)), labels=xticklabels)
    ax.set_yticks(np.arange(len(yticklabels)), labels=yticklabels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # set title and x/y labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Option 1 - Plot matrix with lognorm cmap
    # im = ax.imshow(matrix, cmap=cm.viridis, norm=LogNorm(vmin=0.01, vmax=610))

    # Option 2 - Plot matrix with one color in diagonal and another in second diagonal
    A1 = np.zeros(matrix.shape)
    A2 = np.zeros(matrix.shape)
    A3 = np.zeros(matrix.shape)
    # Diagonal elements
    (rows, cols) = matrix.shape
    n = min(rows, cols)
    A1[range(n), range(n)] = 1
    # Next to diagonal elements
    for i in range(max(rows, cols)-1):
        if i <= rows-1 and i + 1 <= cols-1:
            A2[i, i+1] = 1
        if i+1 <= rows-1 and i <= cols-1:
            A2[i+1, i] = 1
    # Apply a mask to filter out unused values
    A1[A1 == 0] = None
    A2[A2 == 0] = None
    A3[A1 == 1] = None
    A3[A2 == 1] = None

    plt.imshow(A1, cmap=ListedColormap(["#55a868"]))
    plt.imshow(A2, cmap=ListedColormap(["#dd8452"]))
    plt.imshow(A3, cmap=ListedColormap(["#718fc1"]))

    # Calculate sum of diagonal values
    b = np.asarray(matrix)
    print('Diagonal (sum): ', np.trace(b))

    # Calculate sum of next to diagonal values
    value = 0
    for i in range(max(rows, cols)-1):
        if i <= rows-1 and i+1 <= cols-1:
            value += matrix.iloc[i, i+1]
        if i+1 <= rows-1 and i <= cols-1:
            value += matrix.iloc[i+1, i]
    print('Next to diagonal (sum): ', value)

    print('All values (sum): ', matrix.sum().sum())

    plt.text(cols, 0, f'diagonal={np.trace(b)}',
             bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.text(cols, 1, f'next to diagonal={value}',
             bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.text(cols, 2, f'total elements={matrix.sum().sum()}',
             bbox=dict(fill=False, edgecolor='red', linewidth=2))
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir,
                             f'Confusion_matrix_diagonal_{sim_code}.png'))
    plt.close()
    # plt.show()
