import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
import parsers
import os
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
    total_acc = (mat.diagonal().sum() / mat.sum()) * 100
    mat = mat / np.sum(mat, -1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    ax.set_xticklabels(['away', 'left', 'right'])
    ax.set_yticklabels(['away', 'left', 'right'])
    plt.axis('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    logging.info('acc:{:.4f}%'.format(total_acc))
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

def compare_two_coding_files(coding1, coding2):
    """
    given two codings in the format described below, returns some relevant metrics in a dictionary
    format:
    [list of [frame number, valid, class], first annotated frame, last annotated frame]
    :param coding1: the "target"
    :param coding2: the "prediction"
    note: naming the codings as target and prediction is just a convention;
    all functions and metrics are symmetric w.r.t. the codings.
    :return:
    """
    start1 = coding1[1]
    end1 = coding1[2]
    start2 = coding2[1]
    end2 = coding2[2]
    coding1_np = np.array([x[0] for x in coding1[0]])
    coding2_np = np.array([x[0] for x in coding2[0]])

    classes = {"away": 0, "left": 1, "right": 2}
    ON_CLASSES = ["left", "right"]
    same_count = 0
    times_coding1 = {"left": [],
                    "right": [],
                    "away": [],
                    "none": []}

    times_coding2 = {"left": [],
                      "right": [],
                      "away": [],
                      "none": []}

    coding1_by_label = [0, 0, 0]
    coding2_by_label = [0, 0, 0]

    left_right_agree = 0
    c1_left_right_total = 0
    c2_left_right_total = 0

    valid_labels_coding1 = 0
    valid_labels_coding2 = 0
    start = min(start1, start2)
    end = max(end1, end2)
    total_frames_coding1 = end1 - start1
    total_frames_coding2 = end2 - start2
    for frame_index in range(start, end):
        if frame_index < coding1[0][0][0]:
            coding1_label = [None, None, "none"]
            times_coding1["none"].append(frame_index)
        else:
            target_q_np = np.nonzero(frame_index >= coding1_np)[0][-1]
            coding1_label = coding1[0][target_q_np]
            if coding1_label[1] == 1:
                assert coding1_label[2] in classes.keys()
                times_coding1[coding1_label[2]].append(frame_index)
                valid_labels_coding1 += 1
            else:
                coding1_label = [None, None, "none"]
                times_coding1["none"].append(frame_index)
        if frame_index < coding2[0][0][0]:
            coding2_label = [None, None, "none"]
            times_coding2["none"].append(frame_index)
        else:
            inferred_q_np = np.nonzero(frame_index >= coding2_np)[0][-1]
            coding2_label = coding2[0][inferred_q_np]
            if coding2_label[1] == 1:
                assert coding2_label[2] in classes.keys()
                times_coding2[coding2_label[2]].append(frame_index)
                valid_labels_coding2 += 1
            else:
                coding2_label = [None, None, "none"]
                times_coding2["none"].append(frame_index)

        if coding1_label[2] != "none":
            assert coding1_label[2] in classes.keys()
            coding1_by_label[classes[coding1_label[2]]] += 1
        if coding2_label[2] != "none":
            assert coding2_label[2] in classes.keys()
            coding2_by_label[classes[coding2_label[2]]] += 1
        if coding1_label[2] == coding2_label[2] and coding1_label[2] != "none":
            assert coding1_label[2] in classes.keys()
            same_count += 1
        if coding1_label[2] in ON_CLASSES:
            if coding1_label[2] == coding2_label[2]:
                left_right_agree += 1
            c1_left_right_total += 1
        if coding2_label[2] in ON_CLASSES:
            c2_left_right_total += 1

    accuracy_coding1 = 100 * same_count / valid_labels_coding1
    accuracy_coding2 = 100 * same_count / valid_labels_coding2
    num_coding1_valid = 100 * (valid_labels_coding1 / total_frames_coding1)
    num_coding2_valid = 100 * (valid_labels_coding2 / total_frames_coding2)

    coding1_on_vs_away = (coding1_by_label[0] + coding1_by_label[1]) / sum(coding1_by_label)
    coding2_on_vs_away = (coding2_by_label[0] + coding2_by_label[1]) / sum(coding2_by_label)
    c1_left_right_accuracy = left_right_agree / c1_left_right_total
    c2_left_right_accuracy = left_right_agree / c2_left_right_total

    metrics = {"accuracy_coding1": accuracy_coding1,
               "accuracy_coding2": accuracy_coding2,
               "num_coding1_valid": num_coding1_valid,
               "num_coding2_valid": num_coding2_valid,
               "coding1_on_vs_away": coding1_on_vs_away,
               "coding2_on_vs_away": coding2_on_vs_away,
               "left_right_accuracy_c1": c1_left_right_accuracy,
               "left_right_accuracy_c2": c2_left_right_accuracy,
               "coding1_by_label": coding1_by_label,
               "coding2_by_label": coding2_by_label,
               "times_coding1": times_coding1,
               "times_coding2": times_coding2,
               "valid_range_coding1": [start1, end1],
               "valid_range_coding2": [start2, end2]}
    return metrics


def compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args):
    logging.info("comparing target and inferred labels: {target_path} vs {inferred_path}")
    parser1 = parsers.PrefLookTimestampParser(30)
    parser2 = parsers.PrincetonParser(30,
                                      start_time_file="/disk3/yotam/icatcher+/datasets/marchman_raw/Visit_A/start_times_visitA.csv")
    if args.machine_coding_format == "PrefLookTimestamp":
        parser = parser1
    else:
        parser = parser2
    machine, mstart, mend = parser.parse(machine_coding_file, file_is_fullpath=True)

    if args.human_coding_format == "PrefLookTimestamp":
        parser = parser1
    else:
        parser = parser2
    human, start1, end1 = parser.parse(human_coding_file, file_is_fullpath=True)
    human2, start2, end2 = parser.parse(human_coding_file2, file_is_fullpath=True)
    # machine = machine[:-1]
    metrics = {}
    metrics["human1_vs_machine"] = compare_two_coding_files([human, start1, end1], [machine, mstart, mend])
    metrics["human1_vs_human2"] = compare_two_coding_files([human, start1, end1], [human2, start2, end2])
    # total_h1_annotated_frames = end1 - start1
    # total_h2_annotated_frames = end2 - start2
    # total_m_annotated_frames = mend - mstart
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


def get_frame_from_video(ID, time_in_ms, video_folder):
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


def sample_luminance(ID, args, start, end, num_samples=10):
    total_luminance = 0
    sampled = 0
    for video_file in args.raw_video_folder.glob("*"):
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


def generate_frame_comparison(sorted_IDs, all_metrics, args):
    GRAPH_CLASSES = ["left", "right", "away", "none"]
    widths = [8, 1, 1]
    heights = [1] * len(sorted_IDs)
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(len(sorted_IDs), 3, figsize=(30, 45), gridspec_kw=gs_kw)
    plt.suptitle(f'Frame by frame comparison of {" ".join(INFERENCE_METHODS)} and human labels', fontsize=40)
    color_gradient = list(Color("red").range_to(Color("green"), 100))

    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        timeline, accuracy, sample_frame = axs[i, :]  # won't work with single video...

        start1, end1 = all_metrics[target_ID]["human1_vs_machine"]["valid_range_coding1"]
        start2, end2 = all_metrics[target_ID]["human1_vs_machine"]["valid_range_coding2"]
        start = min(start1, start2)
        end = max(end1, end2)
        # end = start + 5000
        timeline.set_title("Video ID: " + str(target_ID.stem) + " (Times " + str(start) + "-" + str(end) + " milliseconds)",
                           fontsize=14)
        for name in INFERENCE_METHODS:
            times = all_metrics[target_ID][name]['times_coding2']
            video_label = str(i) + '-' + name
            skip = 100  # frame comparison resolution. Increase to speed up plotting
            for label in GRAPH_CLASSES:
                timeline.barh(video_label, skip, left=times[label][::skip], height=1, label=label,
                              color=LABEL_TO_COLOR[label])
            timeline.set_xlabel("Time (ms)")

            if i == 0 and name == "machine":
                timeline.legend(loc='upper right')
                accuracy.set_title("Frame by frame accuracy for each model")
        accuracies = [all_metrics[target_ID][inference]['accuracy_coding2'] for inference in INFERENCE_METHODS]
        # colors = [color_gradient[int(acc * 100)].rgb for acc in accuracies]

        accuracy.bar(range(len(INFERENCE_METHODS)), accuracies, color="black")
        accuracy.set_xticks(range(len(INFERENCE_METHODS)))
        accuracy.set_xticklabels(INFERENCE_METHODS)
        accuracy.set_ylim([0, 1])
        accuracy.set_ylabel("Accuracy")
        # sample_frame_index = min(
        #     [all_metrics[target_ID][ICATCHER]['times_target'][label][0] for label in VALID_CLASSES])
        sample_frame_index = (end - start) / 2.0
        sample_frame.imshow(get_frame_from_video(target_ID, sample_frame_index, args.raw_video_folder))
        sample_frame.set_title(f'Sample frame from video at time {int(sample_frame_index)}')
    plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
    plt.savefig(Path(args.output_folder, 'frame_by_frame_all.png'))
    plt.cla()
    plt.clf()


def generate_plot_set(sorted_IDs, all_metrics, inference, save_path):
    classes = {"away": 0, "left": 1, "right": 2}
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
    accuracies = [all_metrics[ID][inference]['accuracy_coding2'] for ID in sorted_IDs]
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

    x_target_valid = [all_metrics[ID][inference]['num_coding1_valid'] for ID in sorted_IDs]
    y_target_valid = [all_metrics[ID][inference]['num_coding2_valid'] for ID in sorted_IDs]
    target_valid_scatter.scatter(x_target_valid, y_target_valid)
    for i in range(len(sorted_IDs)):
        target_valid_scatter.annotate(i, (x_target_valid[i], y_target_valid[i]))
    target_valid_scatter.set_xlabel("Human annotated labels")
    target_valid_scatter.set_xlabel(f'{inference} labels')

    target_valid_scatter.set_title(
        f'Number of distinct look events\n(\"left, right, away\") per second\nfor {inference} vs human data')

    x_target_away = [all_metrics[ID][inference]['coding1_on_vs_away'] for ID in sorted_IDs]
    y_target_away = [all_metrics[ID][inference]['coding2_on_vs_away'] for ID in sorted_IDs]
    on_away_scatter.scatter(x_target_away, y_target_away)
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away[i], y_target_away[i]))

    on_away_scatter.set_xlabel("Human annotated labels")
    on_away_scatter.set_xlabel(f'{inference} labels')
    on_away_scatter.set_title(
        f'Portion of left and right labels compared to\ntotal number of left, right, and away labels\nfor {inference} vs Human data')

    for i, label in enumerate(sorted(classes.keys())):
        x_labels = [y[i] / sum(y) for y in [all_metrics[ID][inference]['coding2_by_label'] for ID in sorted_IDs]]
        y_labels = [y[i] / sum(y) for y in [all_metrics[ID][inference]['coding1_by_label'] for ID in sorted_IDs]]
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
                            [all_metrics[ID][inference]['coding2_by_label'] for ID in sorted_IDs]]
        label_counts_tar = [y[i] / sum(y) for y in [all_metrics[ID][inference]['coding1_by_label'] for ID in sorted_IDs]]

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

    plt.scatter([all_metrics[id]["human1_vs_human2"]["accuracy_coding2"] for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine"]["accuracy_coding2"] for id in sorted_IDs])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Human agreement")
    plt.ylabel(f'{"machine"} accuracy')
    plt.title(f'Inference accuracy versus human agreement for the {len(all_metrics)} doubly coded videos')
    plt.savefig(Path(args.output_folder, 'iCatcher_acc_vs_certainty.png'))
    plt.cla()
    plt.clf()


def plot_luminance_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.scatter([sample_luminance(id, args, *all_metrics[id]["human1_vs_machine"]['valid_range_coding2']) for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine"]["accuracy_coding2"] for id in sorted_IDs])
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
        assert len(coding_intersect) > 0
        all_metrics = {}
        for code_file in coding_intersect:
            human_coding_file = Path(args.human_codings_folder / code_file)
            human_coding_file2 = Path(args.human2_codings_folder / code_file)
            machine_coding_file = Path(args.machine_codings_folder / code_file)
            all_metrics[human_coding_file] = compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args)
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


def make_gallery(array, save_path, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    plt.imshow(result)
    plt.savefig(save_path)
    plt.cla()
    plt.clf()

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
                        key=lambda x: all_metrics[x]["human1_vs_machine"]["accuracy_coding2"])  # sort by accuracy
    # sorted_ids = sorted(list(all_metrics.keys()))  # sort alphabetically
    INFERENCE_METHODS = ["human1_vs_human2", "human1_vs_machine"]
    for inference in INFERENCE_METHODS:
        # save_metrics_csv(sorted_ids, all_metrics, inference)
        generate_plot_set(sorted_ids, all_metrics, inference, args.output_folder)
    generate_frame_comparison(random.sample(sorted_ids, min(len(sorted_ids), 8)), all_metrics, args)
    plot_inference_accuracy_vs_human_agreement(sorted_ids, all_metrics, args)
    plot_luminance_vs_accuracy(sorted_ids, all_metrics, args)
