import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_confusion_matrix(label, pred, save_path, class_num=3):
    mat = np.zeros([class_num, class_num])
    pred = np.array(pred)
    label = np.array(label)
    acc = sum(pred == label) / len(label)
    print('acc:{:.4f}'.format(acc))
    print('# datapoint', len(label))
    for i in range(class_num):
        for j in range(class_num):
            mat[i][j] = sum((label == i) & (pred == j))
    print(mat)
    mat = mat / np.sum(mat, -1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    ax.set_xticklabels(['away', 'left', 'right'])
    ax.set_yticklabels(['away', 'left', 'right'])
    plt.axis('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    return acc


def plot_learning_curve(train_perfs, val_perfs, save_dir, isLoss=False):
  epochs = np.arange(1, len(train_perfs) + 1)
  plt.plot(epochs, train_perfs, label="Training set")
  plt.plot(epochs, val_perfs, label="Validation set")
  plt.xlabel("Epochs")
  metric_name = "Loss" if isLoss else "Accuracy"
  plt.ylabel(metric_name)
  plt.title(metric_name, fontsize=16, y=1.002)
  plt.legend()
  plt.savefig(os.path.join(save_dir, 'learning_curve_%s.png' % metric_name))
  plt.clf()