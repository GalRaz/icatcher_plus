import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
from parsers import PrefLookTimestampParser
import os
from config import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import csv
import logging
import cv2
from colour import Color
from pathlib import Path
from options import parse_arguments_for_visualizations


### todo: retrieve globals from command line arguments ###
DEPLOYMENT_ROOT = Path("/home/jupyter")
GRAPH_CLASSES = ["left", "right", "away", "none"]
ON_CLASSES = ["left", "right"]
OFF_CLASSES = ["away"]
LABEL_TO_COLOR = {"left": (0.5, 0.6, 0.9), "right": (0.6, 0.8, 0), "away": (0.95, 0.5, 0.4), "none": "lightgrey"}
# logging.basicConfig(level=logging.CRITICAL)
###########################################################


def calculate_confusion_matrix(label, pred, save_path, mat=None, class_num=3):
    """
    creates a plot of the confusion matrix given the gt labels abd the predictions.
    if mat is supplied, ignores other inputs and uses that.
    :param label: the labels
    :param pred: the predicitions
    :param save_path: path to save plot
    :param mat: a numpy 3x3 array representing the confusion matrix
    :param class_num: number of classes
    :return:
    """
    if mat is None:
        mat = np.zeros([class_num, class_num])
        pred = np.array(pred)
        label = np.array(label)
        logging.info('# datapoint: {}'.format(len(label)))
        for i in range(class_num):
            for j in range(class_num):
                mat[i][j] = sum((label == i) & (pred == j))
    logging.info("confusion matrix:{}".format(mat))
    mat = mat / np.sum(mat, -1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    ax.set_xticklabels(['away', 'left', 'right'])
    ax.set_yticklabels(['away', 'left', 'right'])
    plt.axis('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    total_acc = mat.diagonal().sum() / class_num
    logging.info('acc:{:.4f}'.format(total_acc))
    return mat, total_acc


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
        plt.cla()
        plt.clf()

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
    plt.cla()
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


def time_ms_to_frame(time_ms, fps=30):
    return int(time_ms / 1000.0 * fps)


def frame_to_time_ms(frame, fps=30):
    return int(frame * 1000 / fps)


def compare_files(human_coding_file, human_coding_file2, machine_coding_file, offset=0):
    logging.info("comparing target and inferred labels: {target_path} vs {inferred_path}")
    # VALID_CLASSES = ["left", "right", "away"]
    parser = PrefLookTimestampParser()

    human, start, end = parser.parse(human_coding_file)
    human_np = np.array([x[0] for x in human])
    human2, _, _ = parser.parse(human_coding_file2)
    human2_np = np.array([x[0] for x in human2])
    machine, _, _ = parser.parse(machine_coding_file)
    machine_np = np.array([x[0] for x in machine])
    machine = machine[:-1]
    codings = {"machine": [machine, machine_np],
               "human2": [human2, human2_np]}

    total_frames = end - start
    metrics = dict()

    for coding in codings:
        annotation = codings[coding][0]
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

        logging.info(f'There are {len(human)} labels in the human codings')
        logging.info(f'There are {len(annotation)} labels in the {coding} codings')

        valid_labels_target = set()
        valid_labels_inferred = set()
        for frame_index in range(start, end):
            if frame_index >= human[0][0]:  # wait for first ground truth (target) label
                target_q_np = np.nonzero(frame_index >= human_np)[0][-1]
                # target_q = [index for index, val in enumerate(human) if frame_index >= val[0]]
                # assert target_q_np==max(target_q)
                target_label = human[target_q_np]
                if frame_index >= annotation[0][0]:  # get the inferred label if it exists
                    inferred_q_np = np.nonzero(frame_index >= (codings[coding][1] - offset))[0][-1]
                    # inferred_q = [index for index, val in enumerate(annotation) if frame_index >= val[0] - offset]
                    # assert inferred_q_np==max(inferred_q)
                    inferred_label = annotation[inferred_q_np]
                else:
                    inferred_label = [0, 0, "none"]

                if target_label[2] in classes.keys():
                    times_target[target_label[2]].append(frame_index)
                else:
                    times_target["none"].append(frame_index)

                if inferred_label[2] in classes.keys():
                    times_inferred[inferred_label[2]].append(frame_index)
                else:
                    times_inferred["none"].append(frame_index)

                # only consider frames where the target label is valid:
                if target_label[1] == 1:
                    if target_label[2] not in classes.keys():
                        logging.critical("thats weird: {target_label}")
                    valid_labels_target.add(target_label[0])
                    valid_labels_inferred.add(inferred_label[0])

                    valid_frames_target += 1
                    if target_label[2] == inferred_label[2]:
                        same_count += 1

                    if target_label[2] in classes.keys():
                        target_by_label[classes[target_label[2]]] += 1
                    if inferred_label[2] in classes.keys():

                        inferred_by_label[classes[inferred_label[2]]] += 1

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

        metrics[coding] = {"accuracy": accuracy,
                           "num_target_valid": num_target_valid,
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
        csv_writer = csv.writer(csvfile, delimiter=',')
        header = ["video ID", f'{inference} label path', "Accuracy"]
        csv_writer.writerow(header)
        for ID in sorted_IDs:
            row = []
            row.append(ID)
            bucket_root = "gaze-coding/iCatcher/pre-trained-inference/"
            row.append(bucket_root + all_metrics[ID][inference]['filename'])
            row.append(all_metrics[ID][inference]['accuracy'])
            csv_writer.writerow(row)


def get_frame_from_video(ID, time_in_ms):
    for video_file in Path(video_folder).glob("*"):
        if ID.stem in video_file.name:
            cap = cv2.VideoCapture(str(video_file))
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
    for video_file in video_folder.glob("*"):
        if ID.stem in video_file.stem:
            cap = cv2.VideoCapture(str(video_file))
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


def generate_frame_comparison(sorted_IDs, all_metrics, save_path):
    widths = [8, 1, 1]
    heights = [1] * len(sorted_IDs)
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    fig, axs = plt.subplots(len(sorted_IDs), 3, figsize=(30, 45), gridspec_kw=gs_kw)
    plt.suptitle(f'Frame by frame comparison of {" ".join(INFERENCE_METHODS)} and human labels', fontsize=40)
    color_gradient = list(Color("red").range_to(Color("green"), 100))

    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        timeline, accuracy, sample_frame = axs

        start, end = all_metrics[target_ID]["machine"]["valid_range"]
        # end = start + 5000
        timeline.set_title("Video ID: " + str(target_ID) + " (Times " + str(start) + "-" + str(end) + " milliseconds)",
                           fontsize=14)
        for name in ["coding_first", "coding_second", "machine"]:
            if name in INFERENCE_METHODS:  # iCatcher or openGaze
                times = all_metrics[target_ID][name]['times_inferred']
            else:  # human data
                times = all_metrics[target_ID]["machine"]['times_target']

            video_label = str(i) + '-' + name
            skip = 100  # frame comparison resolution. Increase to speed up plotting
            for label in GRAPH_CLASSES:
                timeline.barh(video_label, skip, left=times[label][::skip], height=1, label=label,
                              color=LABEL_TO_COLOR[label])
            timeline.set_xlabel("Time (ms)")

            if i == 0 and name == "machine":
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
    plt.savefig(Path(save_path, 'frame_by_frame_all.png'))
    plt.cla()
    plt.clf()


def generate_plot_set(sorted_IDs, all_metrics, inference, save_path):
    # VALID_CLASSES = ["left", "right", "away"]
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

    for i, label in enumerate(sorted(classes.keys())):
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
    for i, label in enumerate(sorted(classes.keys())):
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
    plt.savefig(Path(save_path, "{}.png".format(inference)))
    plt.cla()
    plt.clf()

def plot_inference_accuracy_vs_human_agreement(sorted_IDs, all_metrics, args):

    plt.scatter([all_metrics[id]["human2"]["accuracy"] for id in sorted_IDs],
                [all_metrics[id]["machine"]["accuracy"] for id in sorted_IDs])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Human agreement")
    plt.ylabel(f'{"machine"} accuracy')
    plt.title(f'Inference accuracy versus human agreement for the {len(all_metrics)} doubly coded videos')
    plt.savefig(Path(args.output_folder, 'iCatcher_acc_vs_certainty.png'))
    plt.cla()
    plt.clf()

def plot_luminance_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.scatter([sample_luminance(id, *all_metrics[id]["machine"]['valid_range']) for id in sorted_IDs],
                [all_metrics[id]["machine"]["accuracy"] for id in sorted_IDs])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel("Luminance")
    plt.ylabel(f'{"machine"} accuracy')
    plt.title(f'Inference accuracy versus mean video luminance for {len(all_metrics)} doubly coded videos')
    plt.savefig(Path(args.output_folder, 'iCatcher_lum_vs_acc.png'))
    plt.cla()
    plt.clf()

def create_cache_metrics(args, force_create=False):
    """
    creates cache that can be used instead of parsing the annotation files from scratch each time
    :return:
    """
    metric_save_path = Path(args.output_folder, "metrics.p")
    if metric_save_path.is_file() and not force_create:
        all_metrics = pickle.load(open(metric_save_path, "rb"))
    else:
        machine_annotation = []
        human_annotation = []
        human_annotation2 = []
        # target_root = label_folder
        # coding_first = set([x.name for x in Path(label_folder).glob("*")])
        # coding_second = set([x.name for x in Path(label2_folder).glob("*")])
        # coding_intersect = coding_first.intersection(coding_second)

        # Get a list of all machine annotation files
        for file in Path(args.human_codings_folder).glob("*"):
            human_annotation.append(file.name)
        for file in Path(args.human2_codings_folder).glob("*"):
            human_annotation2.append(file.name)
        for file in Path(args.machine_codings_folder).glob("*"):
            machine_annotation.append(file.name)
        coding_intersect = set(human_annotation2).intersection(set(human_annotation))
        coding_intersect = coding_intersect.intersection(set(machine_annotation))
        # for subdir, dirs, files in os.walk(annotation_folder):
        #     for filename in files:
        #         machine_annotation.append(os.path.abspath(annotation_folder / filename))

        # Get a list of all human annotation files
        # for subdir, dirs, files in os.walk(target_root):
        #     for filename in files:
        #         logging.info("found label file {target_root}{filename}")
        #         if filename in coding_intersect:
        #             human_annotation.append(target_root / filename)

        # sort the file paths alphabetically to pair them up

        coding_intersect = sorted(list(coding_intersect))
        all_metrics = {}
        for code_file in coding_intersect:
            human_coding_file = Path(args.human_codings_folder / code_file)
            human_coding_file2 = Path(args.human2_codings_folder / code_file)
            machine_coding_file = Path(args.machine_codings_folder / code_file)
            all_metrics[human_coding_file] = compare_files(human_coding_file, human_coding_file2, machine_coding_file)
            # all_metrics[human_coding_file][ICATCHER_PLUS]['filename'] = machine_coding_file.name


        # all_metrics = {}
        # for i, h_coding in enumerate(tqdm(human_annotation)):
        #     for j, m_coding in enumerate(machine_annotation):
        #         human_coding_id = h_coding.stem
        #         machine_coding_id = m_coding.stem
        #         if machine_coding_id in human_coding_id:
        #             all_metrics[human_coding_id] = compare_files(h_coding, m_coding)
        #             all_metrics[human_coding_id][ICATCHER_PLUS]['filename'] = m_coding.name
        #             break

        # Store the intermediate results so we can access them without regenerating everything:
        pickle.dump(all_metrics, open(metric_save_path, "wb"))
    return all_metrics


def put_text(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img, class_name,
                top_left_corner_text,
                font,
                font_scale,
                font_color,
                line_type)
    return img


def put_arrow(img, class_name, face):
    arrow_start_x = int(face[0] + 0.5 * face[2])
    arrow_end_x = int(face[0] + 0.1 * face[2] if class_name == "left" else face[0] + 0.9 * face[2])
    arrow_y = int(face[1] + 0.8 * face[3])
    img = cv2.arrowedLine(img,
                          (arrow_start_x, arrow_y),
                          (arrow_end_x, arrow_y),
                          (0, 255, 0),
                          thickness=3,
                          tipLength=0.4)
    return img


def put_rectangle(popped_frame, face):
    color = (0, 255, 0)  # green
    thickness = 2
    popped_frame = cv2.rectangle(popped_frame,
                                 (face[0], face[1]), (face[0] + face[2], face[1] + face[3]),
                                 color,
                                 thickness)
    return popped_frame


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


if __name__ == "__main__":
    args = parse_arguments_for_visualizations()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())

    # args.human_codings_folder = Path("/disk3/yotam/icatcher+/datasets/lookit/coding_first")
    # args.human2_codings_folder = Path("/disk3/yotam/icatcher+/datasets/lookit/coding_second")
    # args.machine_codings_folder = Path("/disk3/yotam/icatcher+/datasets/lookit/coding_machine")
    # args.output_folder = Path("/disk3/yotam/icatcher+/runs/vanilla/output")

    all_metrics = create_cache_metrics(args, force_create=False)
    sorted_ids = sorted(list(all_metrics.keys()),
                        key=lambda x: all_metrics[x]["machine"]["accuracy"])  # sort by accuracy
    # sorted_ids = sorted(list(all_metrics.keys()))  # sort alphabetically
    INFERENCE_METHODS = ["human2", "machine"]
    for inference in INFERENCE_METHODS:
        # save_metrics_csv(sorted_ids, all_metrics, inference)
        generate_plot_set(sorted_ids, all_metrics, inference, args.output_folder)
    generate_frame_comparison(random.sample(sorted_ids, min(len(sorted_ids), 8)), all_metrics, args.output_folder)
    plot_inference_accuracy_vs_human_agreement(sorted_ids, all_metrics, args)
    plot_luminance_vs_accuracy(sorted_ids, all_metrics, args)
