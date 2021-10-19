import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from config import *
from sklearn.metrics import confusion_matrix


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


def confusion_mat(targets, preds, classes, normalize=False, plot=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    cm = confusion_matrix(targets, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(title + ".png")
        plt.show()

    return cm


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


def print_data_img_name(dataloaders, dl_key, selected_idxs=None):
    dataset = dataloaders[dl_key].dataset
    fp_prefix = os.path.join(face_data_folder, dl_key)
    if selected_idxs is None:
        for fname, lbl in dataset.samples:
            print(fname.strip(fp_prefix))
    else:
        for i, tup in enumerate(dataset.samples):
            if i in selected_idxs:
                print(Path(tup[0]).parent.stem, Path(tup[0]).stem)


import sys

from parsers import MyParserTxt
import os
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import csv
import logging
import cv2

from colour import Color

DEPLOYMENT_ROOT = Path("/home/jupyter")

VALID_CLASSES = ["left", "right", "away"]
GRAPH_CLASSES = ["left", "right", "away", "none"]
ON_CLASSES = ["left", "right"]
OFF_CLASSES = ["away"]
LABEL_TO_COLOR = {"left": (0.5, 0.6, 0.9), "right": (0.6, 0.8, 0), "away": (0.95, 0.5, 0.4), "none": "lightgrey"}
ICATCHER_PLUS = "iCatcher+"
CODING_FIRST = "coding_first"
CODING_SECOND = "coding_second"
INFERENCE_METHODS = [CODING_SECOND, ICATCHER_PLUS]
logging.basicConfig(level=logging.CRITICAL)
METRIC_SAVE_PATH = visualization_root / "iCatcherPlus.p"


def time_ms_to_frame(time_ms, fps=30):
    return int(time_ms / 1000.0 * fps)


def frame_to_time_ms(frame, fps=30):
    return int(frame * 1000 / fps)


# def get_open_gaze_label(time_ms, ID):
#     for video_path, label_list in OPEN_GAZE_LABELS.items():
#         if ID in video_path:
#             try:
#                 return [time_ms, 0, label_list[time_ms_to_frame(time_ms)]]
#             except IndexError:
#                 return [time_ms, "none", 0]
#     raise ValueError("ID not found in opengaze labels")

# def parse_open_gaze(ID):
#     labels = []
#     for video_path, label_list in OPEN_GAZE_LABELS.items():
#         if ID in video_path:
#             for frame_idx, label in enumerate(label_list):
#
#                 if frame_idx == 0:
#                     labels.append([0, 1, label])
#                 else:
#                     if label != labels[-1][2]:
#                         labels.append([frame_to_time_ms(frame_idx), 1, label])
#             return labels
#     return
#     raise ValueError("ID not found in opengaze labels")


def compare_files(target_path, inferred_path, ID, offset=0):
    logging.info("comparing target and inferred labels: {target_path} vs {inferred_path}")
    parser = MyParserTxt()

    target, start, end = parser.parse(target_path)
    coding_second, _, _ = parser.parse(str(target_path).replace("first", "second"))
    i_catcher_parsed = parser.parse(inferred_path)[0][:-1]
    labels = {ICATCHER_PLUS: i_catcher_parsed,
              CODING_SECOND: coding_second}

    total_frames = end - start
    metrics = dict()

    for inference in INFERENCE_METHODS:

        inferred = labels[inference]
        same_count = 0
        valid_frames_target = 0

        times_target = {"left": [],
                        "right": [],
                        "away": [],
                        "none": []}

        times_inferred = {"left": [],
                          "right": [],
                          "away": [],
                          "none": []}

        target_by_label = [0, 0, 0]
        inferred_by_label = [0, 0, 0]

        left_right_agree = 0
        left_right_total = 0

        logging.info(f'There are {len(target)} labels in the human annotated data')
        logging.info(f'There are {len(inferred)} labels in the {inference} output')

        valid_labels_target = set()
        valid_labels_inferred = set()
        for frame_index in range(start, end):
            if frame_index >= target[0][0]:  # wait for first ground truth (target) label
                target_q = [index for index, val in enumerate(target) if frame_index >= val[0]]
                target_label = target[max(target_q)]
                if frame_index >= inferred[0][0]:  # get the inferred label if it exists
                    inferred_q = [index for index, val in enumerate(inferred) if frame_index >= val[0] - offset]
                    inferred_label = inferred[max(inferred_q)]
                else:
                    inferred_label = [0, 0, "none"]

                if target_label[2] in VALID_CLASSES:
                    times_target[target_label[2]].append(frame_index)
                else:
                    times_target["none"].append(frame_index)

                if inferred_label[2] in VALID_CLASSES:
                    times_inferred[inferred_label[2]].append(frame_index)
                else:
                    times_inferred["none"].append(frame_index)

                # only consider frames where the target label is valid:
                if target_label[1] == 1:
                    if target_label[2] not in VALID_CLASSES:
                        logging.critical("thats weird: {target_label}")
                    valid_labels_target.add(target_label[0])
                    valid_labels_inferred.add(inferred_label[0])

                    valid_frames_target += 1
                    if target_label[2] == inferred_label[2]:
                        same_count += 1

                    if target_label[2] in VALID_CLASSES:
                        target_by_label[VALID_CLASSES.index(target_label[2])] += 1
                    if inferred_label[2] in VALID_CLASSES:
                        inferred_by_label[VALID_CLASSES.index(inferred_label[2])] += 1

                    if target_label[2] in ON_CLASSES:
                        if target_label[2] == inferred_label[2]:
                            left_right_agree += 1
                        left_right_total += 1

        accuracy = same_count / valid_frames_target
        num_target_valid = len(valid_labels_target) / total_frames * 1000
        num_inferred_valid = len(valid_labels_inferred) / total_frames * 1000

        target_on_vs_away = (target_by_label[0] + target_by_label[1]) / sum(target_by_label)
        inferred_on_vs_away = (inferred_by_label[0] + inferred_by_label[1]) / sum(target_by_label)
        left_right_accuracy = left_right_agree / left_right_total

        metrics[inference] = {"accuracy": accuracy,
                              "num_target_valid": num_target_valid,
                              "num_inferred_valid": num_inferred_valid,
                              "num_inferred_valid": num_inferred_valid,
                              "target_on_vs_away": target_on_vs_away,
                              "inferred_on_vs_away": inferred_on_vs_away,
                              "left_right_accuracy": left_right_accuracy,
                              "inferred_by_label": inferred_by_label,
                              "target_by_label": target_by_label,
                              "times_target": times_target,
                              "times_inferred": times_inferred,
                              "valid_range": [start, end]}

    return metrics


def save_metrics_csv(sorted_IDs, all_metrics, inference):
    with open(f'iCatcher/plots/CSV_reports/{inference}', 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        header = ["video ID", f'{inference} label path', "Accuracy"]
        csvWriter.writerow(header)
        for ID in sorted_IDs:
            row = []
            row.append(ID)
            bucket_root = "gaze-coding/iCatcher/pre-trained-inference/"
            row.append(bucket_root + all_metrics[ID][inference]['filename'])
            row.append(all_metrics[ID][inference]['accuracy'])
            csvWriter.writerow(row)


def get_frame_from_video(ID, time_in_ms):
    for subdir, dirs, files in os.walk(video_folder):
        for filename in files:
            if ID in filename:
                video_path = video_folder / filename
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_no = int(time_in_ms / 1000 * fps)
                cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_no - 1)
                ret, frame = cap.read()
                x = 0
                for i in range(frame_no):
                    x = i
                    ret, frame = cap.read()
                if ret:
                    # first convert color mode to RGB
                    b, g, r = cv2.split(frame)
                    rgb_img = cv2.merge([r, g, b])
                    return rgb_img
    return np.ones(shape=(255, 255))


def sample_luminance(ID, start, end, num_samples=10):
    total_luminance = 0
    sampled = 0
    for subdir, dirs, files in os.walk(video_folder):
        for filename in files:
            if ID in filename:
                video_path = video_folder + filename
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_no = int(start / 1000 * fps)

                for i in range(num_samples):
                    cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_no - 1)
                    ret, frame = cap.read()
                    b, g, r = cv2.split(frame)

                    total_luminance += 0.2126 * np.sum(r) + 0.7152 * np.sum(g) + 0.0722 * np.sum(b)
                    sampled += 1
                    frame_no += (((end - start) / num_samples) / 1000 * fps)

    return total_luminance / sampled


def generate_frame_comparison(sorted_IDs, all_metrics):
    widths = [8, 1, 1]
    heights = [1] * len(sorted_IDs)
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    fig, axs = plt.subplots(len(sorted_IDs), 3, figsize=(30, 45), gridspec_kw=gs_kw)
    plt.suptitle(f'Frame by frame comparison of {" ".join(INFERENCE_METHODS)} and human labels', fontsize=40)
    color_gradient = list(Color("red").range_to(Color("green"), 100))

    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        timeline, accuracy, sample_frame = axs[i, :]

        start, end = all_metrics[target_ID][ICATCHER_PLUS]["valid_range"]
        # end = start + 5000
        timeline.set_title("Video ID: " + str(target_ID) + " (Times " + str(start) + "-" + str(end) + " milliseconds)",
                           fontsize=14)
        for name in ["coding_first", "coding_second", ICATCHER_PLUS]:
            if name in INFERENCE_METHODS:  # iCatcher or openGaze
                times = all_metrics[target_ID][name]['times_inferred']
            else:  # human data
                times = all_metrics[target_ID][ICATCHER_PLUS]['times_target']

            video_label = str(i) + '-' + name
            skip = 100
            for label in GRAPH_CLASSES:
                timeline.barh(video_label, skip, left=times[label][::skip], height=1, label=label,
                              color=LABEL_TO_COLOR[label])
            timeline.set_xlabel("Time (ms)")

            if i == 0 and name == ICATCHER_PLUS:
                timeline.legend(loc='upper right')
                accuracy.set_title("Frame by frame accuracy for each model")
        accuracies = [all_metrics[target_ID][inference]['accuracy'] for inference in INFERENCE_METHODS]
        # colors = [color_gradient[int(acc * 100)].rgb for acc in accuracies]

        accuracy.bar(range(len(INFERENCE_METHODS)), accuracies, color="black")
        accuracy.set_xticks(range(len(INFERENCE_METHODS)))
        accuracy.set_xticklabels(INFERENCE_METHODS)
        accuracy.set_ylim([0, 1])
        accuracy.set_ylabel("Accuracy")
        # sample_frame_index = min(
        #     [all_metrics[target_ID][ICATCHER]['times_target'][label][0] for label in VALID_CLASSES])
        sample_frame_index = (end - start) / 2.0
        sample_frame.imshow(get_frame_from_video(target_ID, sample_frame_index))
        sample_frame.set_title(f'Sample frame from video at time {int(sample_frame_index)}')
    plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
    plt.savefig('/home/jupyter/frame_by_frame_all.png')


def generate_plot_set(sorted_IDs, all_metrics, inference):
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    accuracy_bar = axs[0, 0]
    axs[0, 1].axis('off')
    target_valid_scatter = axs[1, 0]
    on_away_scatter = axs[1, 1]
    label_scatter = axs[2, 0]
    label_bar = axs[2, 1]
    scatter_plots = [target_valid_scatter, on_away_scatter, label_scatter]
    for ax in scatter_plots:
        if ax == target_valid_scatter:
            #             pass
            ax.set_xlim([0, 3])
            ax.set_ylim([0, 3])
        else:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", label="Ideal trend")

    ticks = range(len(sorted_IDs))
    x = np.arange(len(sorted_IDs))

    labels = range(len(sorted_IDs))
    ticks = range(len(sorted_IDs))
    accuracies = [all_metrics[ID][inference]['accuracy'] for ID in sorted_IDs]
    accuracy_bar.bar(x, accuracies, color='purple')
    mean = np.mean(accuracies)
    accuracy_bar.axhline(y=mean, color='black', linestyle='-', label="mean (" + str(mean)[:4] + ")")

    accuracy_bar.set_title('Video accuracy (over valid frames)')
    accuracy_bar.set_ylim([0, 1])
    accuracy_bar.set_xticks(ticks)
    accuracy_bar.set_xticklabels(labels, rotation='vertical', fontsize=8)
    accuracy_bar.set_xlabel('Video')

    accuracy_bar.legend()

    # ID_index = axs[0, 1]
    # cell_text = [[i, sorted_IDs[i]] for i in range(len(sorted_IDs))]
    # ID_index.table(cell_text, loc='center', fontsize=18)
    # ID_index.set_title("Video index to ID")

    x_target_valid = [all_metrics[ID][inference]['num_target_valid'] for ID in sorted_IDs]
    y_target_valid = [all_metrics[ID][inference]['num_inferred_valid'] for ID in sorted_IDs]
    target_valid_scatter.scatter(x_target_valid, y_target_valid)
    for i in range(len(sorted_IDs)):
        target_valid_scatter.annotate(i, (x_target_valid[i], y_target_valid[i]))
    target_valid_scatter.set_xlabel("Human annotated labels")
    target_valid_scatter.set_xlabel(f'{inference} labels')

    target_valid_scatter.set_title(
        f'Number of distinct look events\n(\"left, right, away\") per second\nfor {inference} vs human data')

    x_target_away = [all_metrics[ID][inference]['target_on_vs_away'] for ID in sorted_IDs]
    y_target_away = [all_metrics[ID][inference]['inferred_on_vs_away'] for ID in sorted_IDs]
    on_away_scatter.scatter(x_target_away, y_target_away)
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away[i], y_target_away[i]))

    on_away_scatter.set_xlabel("Human annotated labels")
    on_away_scatter.set_xlabel(f'{inference} labels')
    on_away_scatter.set_title(
        f'Portion of left and right labels compared to\ntotal number of left, right, and away labels\nfor {inference} vs Human data')

    for i, label in enumerate(VALID_CLASSES):
        x_labels = [y[i] / sum(y) for y in [all_metrics[ID][inference]['inferred_by_label'] for ID in sorted_IDs]]
        y_labels = [y[i] / sum(y) for y in [all_metrics[ID][inference]['target_by_label'] for ID in sorted_IDs]]
        label_scatter.scatter(x_labels, y_labels, color=LABEL_TO_COLOR[label], label=label)
        for n in range(len(sorted_IDs)):
            label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    label_scatter.set_xlabel(f'{inference} label proportion')
    label_scatter.set_ylabel('human annotated\nlabel proportion')
    label_scatter.set_title(
        f'Normalized number of left, right, and away\nevents for human data vs {inference} inference')
    label_scatter.legend(loc='upper center')

    bottoms_inf = np.zeros(shape=(len(sorted_IDs)))
    bottoms_tar = np.zeros(shape=(len(sorted_IDs)))
    for i, label in enumerate(VALID_CLASSES):
        label_counts_inf = [y[i] / sum(y) for y in
                            [all_metrics[ID][inference]['inferred_by_label'] for ID in sorted_IDs]]
        label_counts_tar = [y[i] / sum(y) for y in [all_metrics[ID][inference]['target_by_label'] for ID in sorted_IDs]]

        label_bar.bar(x - 0.22, label_counts_inf, bottom=bottoms_inf, width=0.4, label=label,
                      color=LABEL_TO_COLOR[label])
        label_bar.bar(x + 0.22, label_counts_tar, bottom=bottoms_tar, width=0.4, label=label,
                      color=LABEL_TO_COLOR[label])

        bottoms_inf += label_counts_inf
        bottoms_tar += label_counts_tar
    label_bar.xaxis.set_major_locator(MultipleLocator(1))
    label_bar.set_xticks(ticks)
    label_bar.set_title('Portion of each label\ntype for each video\n(left is inferred)')
    label_bar.set_ylabel('Portion of total')
    label_bar.set_xlabel('Video')
    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.2, hspace=0.5)
    plt.suptitle(f'{inference} evaluation', fontsize=24)
    plt.savefig(visualization_root / f'{inference}.png')


def plot_inference_accuracy_vs_human_agreement(sorted_IDs, all_metrics):
    plt.scatter([all_metrics[id][CODING_SECOND]["accuracy"] for id in sorted_IDs],
                [all_metrics[id][ICATCHER_PLUS]["accuracy"] for id in sorted_IDs])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Human agreement")
    plt.ylabel(f'{ICATCHER_PLUS} accuracy')
    plt.title(f'Inference accuracy versus human agreement for the {len(all_metrics)} doubly coded videos')
    plt.savefig(visualization_root / 'iCatcher_acc_vs_certainty.png')


def plot_luminance_vs_accuracy(sorted_IDs, all_metrics):
    plt.scatter([sample_luminance(id, *all_metrics[id][ICATCHER_PLUS]['valid_range']) for id in sorted_IDs],
                [all_metrics[id][ICATCHER_PLUS]["accuracy"] for id in sorted_IDs])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel("Luminance")
    plt.ylabel(f'{ICATCHER_PLUS} accuracy')
    plt.title(f'Inference accuracy versus mean video luminance for {len(all_metrics)} doubly coded videos')
    plt.savefig(visualization_root / 'iCatcher_lum_vs_acc.png')


def regenerate_saved_metrics():
    inferreds = []
    targets = []
    inferred_root = inference_output / "iCatcherPlus/training_labels/"

    target_root = label_folder
    coding_first = set([f for s, d, f in os.walk(label_folder)][0])
    coding_second = set([f for s, d, f in os.walk(label2_folder)][0])
    coding_intersect = coding_first.intersection(coding_second)

    # Get a list of all iCatcher data files
    for subdir, dirs, files in os.walk(inferred_root):
        for filename in files:
            if "checkpoint" not in filename:
                inferreds.append(os.path.abspath(inferred_root / filename))

    # Get a list of all target (human annotated) data files
    for subdir, dirs, files in os.walk(target_root):
        for filename in files:
            logging.info("found label file {target_root}{filename}")
            if "checkpoint" not in filename:
                if filename in coding_intersect:
                    targets.append(target_root / filename)

    # sort the file paths alphabetically to pair them up
    targets.sort()
    inferreds.sort()
    # targets = targets[:2] #Uncomment this to pilot a new plot type
    all_metrics = {}
    # regenerate = False
    # if regenerate:
    for i, target in enumerate(tqdm(targets)):
        for j, inferred in enumerate(inferreds):
            target_id = str(target).split("/")[-1].split("-")[0]
            inferred_id = str(inferred).split("/")[-1].split("_")[1]
            if inferred_id in str(target):
                all_metrics[target_id] = compare_files(target, inferred, target_id)
                all_metrics[target_id][ICATCHER_PLUS]['filename'] = inferred.split("/")[-1]
                break

    # Store the intermediate results so we can access them without regenerating everything:
    pickle.dump(all_metrics, open(METRIC_SAVE_PATH, "wb"))


if __name__ == "__main__":
    regenerate_saved_metrics()  # uncomment if you made any changes to how metrics are calculated!

    all_metrics = pickle.load(open(METRIC_SAVE_PATH, "rb"))
    sorted_ids = sorted(list(all_metrics.keys()),
                        key=lambda x: all_metrics[x][ICATCHER_PLUS]['accuracy'])  # sort by accuracy
    # sorted_ids = sorted(list(all_metrics.keys()))  # sort alphabetically
    for inference in INFERENCE_METHODS:
        # save_metrics_csv(sorted_ids, all_metrics, inference)
        generate_plot_set(sorted_ids, all_metrics, inference)
    generate_frame_comparison(random.sample(sorted_ids, min(len(sorted_ids), 8)), all_metrics)
    plot_inference_accuracy_vs_human_agreement(sorted_ids, all_metrics)
    plot_luminance_vs_accuracy(sorted_ids, all_metrics)
    print(all_metrics)
