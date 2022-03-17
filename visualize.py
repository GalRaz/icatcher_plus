import os
from pathlib import Path
import pickle
import itertools
import csv
import preprocess
import logging
from tqdm import tqdm
import numpy as np
import cv2
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.patches import Patch
import textwrap
from options import parse_arguments_for_visualizations
import parsers


def label_to_color(label):
    mapping = {"left": (0.5, 0.6, 0.9),
               "right": (0.9, 0.6, 0.5),
               "away": "lightgrey",
               "invalid": "white",
               "lblue": (0.5, 0.6, 0.9),
               "lred": (0.9, 0.6, 0.5),
               "lgreen": (0.6, 0.8, 0.0),
               "lorange": (0.94, 0.78, 0.0),
               "lyellow": (0.9, 0.9, 0.0),
               "mblue": (0.12, 0.41, 0.87)}
    return mapping[label]


def calculate_confusion_matrix(label, pred, save_path=None, mat=None, class_num=3, verbose=True):
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
        if verbose:
            logging.info('# datapoint: {}'.format(len(label)))
        for i in range(class_num):
            for j in range(class_num):
                mat[i][j] = sum((label == i) & (pred == j))
    if np.all(np.sum(mat, -1, keepdims=True) != 0):
        total_acc = (mat.diagonal().sum() / mat.sum()) * 100
        norm_mat = mat / np.sum(mat, -1, keepdims=True)
    else:
        total_acc = 0
        norm_mat = mat
    if save_path:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax = sns.heatmap(norm_mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
        ax.set_xticklabels(['away', 'left', 'right'])
        ax.set_yticklabels(['away', 'left', 'right'])
        plt.axis('equal')
        plt.tight_layout(pad=0.1)
        plt.savefig(save_path)
    if verbose:
        logging.info('acc:{:.4f}%'.format(total_acc))
        logging.info('confusion matrix: {}'.format(mat))
        logging.info('normalized confusion matrix: {}'.format(norm_mat))
    return norm_mat, mat, total_acc


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


def get_stats_in_interval(start, end, coding1, coding2):
    """
    given two codings (single dimensional numpy arrays) and a start and end time,
    calculates various metrics we care about. assumes coding1[i], coding2[i] refer to same time
    :param start: start time of interval
    :param end: end time of interval
    :param coding1: np array (1 dimensional)
    :param coding2: np array (1 dimensional)
    :return:
    """
    coding1_interval = coding1[start:end]
    coding2_interval = coding2[start:end]
    mutually_valid_frames = np.logical_and(coding1_interval >= 0, coding2_interval >= 0)
    coding1_interval_mut_valid = coding1_interval[mutually_valid_frames]
    coding2_interval_mut_valid = coding2_interval[mutually_valid_frames]

    coding1_away = np.where(coding1_interval == 0)[0]
    coding2_away = np.where(coding2_interval == 0)[0]
    coding1_left = np.where(coding1_interval == 1)[0]
    coding2_left = np.where(coding2_interval == 1)[0]
    coding1_right = np.where(coding1_interval == 2)[0]
    coding2_right = np.where(coding2_interval == 2)[0]
    coding1_invalid = np.where(coding1_interval < 0)[0]
    coding2_invalid = np.where(coding2_interval < 0)[0]


    n_transitions_1 = np.count_nonzero(np.diff(coding1_interval[mutually_valid_frames]))
    n_transitions_2 = np.count_nonzero(np.diff(coding2_interval[mutually_valid_frames]))

    on_screen_1 = np.sum(coding1_interval_mut_valid == 1) + np.sum(coding1_interval_mut_valid == 2)
    on_screen_2 = np.sum(coding2_interval_mut_valid == 1) + np.sum(coding2_interval_mut_valid == 2)
    off_screen_1 = np.sum(coding1_interval_mut_valid == 0)
    off_screen_2 = np.sum(coding2_interval_mut_valid == 0)

    if on_screen_1 == 0:
        percent_r_1 = 0
    else:
        percent_r_1 = np.sum(coding1_interval_mut_valid == 2) / on_screen_1
    if on_screen_2 == 0:
        percent_r_2 = 0
    else:
        percent_r_2 = np.sum(coding2_interval_mut_valid == 2) / on_screen_2

    looking_time_1 = on_screen_1
    looking_time_2 = on_screen_2

    equal = coding1_interval == coding2_interval

    equal_and_non_equal = np.sum(equal[mutually_valid_frames]) + np.sum(np.logical_not(equal[mutually_valid_frames]))
    if equal_and_non_equal == 0:
        agreement = 0
    else:
        agreement = np.sum(equal[mutually_valid_frames]) / equal_and_non_equal
    _, mat, _ = calculate_confusion_matrix(coding1_interval_mut_valid, coding2_interval_mut_valid, verbose=False)
    times_coding1 = {"away": coding1_away,
                     "left": coding1_left,
                     "right": coding1_right,
                     "invalid": coding1_invalid}
    times_coding2 = {"away": coding2_away,
                     "left": coding2_left,
                     "right": coding2_right,
                     "invalid": coding2_invalid}
    return {"n_frames_in_interval": end - start,
            "mutual_valid_frame_count": np.sum(mutually_valid_frames),
            "valid_frames_1": np.sum(coding1_interval >= 0),
            "valid_frames_2": np.sum(coding2_interval >= 0),
            "n_transitions_1": n_transitions_1,
            "n_transitions_2": n_transitions_2,
            "percent_r_1": percent_r_1,
            "percent_r_2": percent_r_2,
            "looking_time_1": looking_time_1,
            "looking_time_2": looking_time_2,
            "agreement": agreement,
            "confusion_matrix": mat,
            "start": start,
            "end": end,
            "times_coding1": times_coding1,
            "times_coding2": times_coding2,
            "label_count_1": [np.sum(coding1_away), np.sum(coding1_left), np.sum(coding1_right), np.sum(coding1_invalid)],
            "label_count_2": [np.sum(coding2_away), np.sum(coding2_left), np.sum(coding2_right), np.sum(coding2_invalid)]
            }


def compare_uncollapsed_coding_files(coding1, coding2, intervals):
    """
    computes various metrics between two codings on a set of intervals
    :param coding1: first coding, uncollapsed numpyarray of events
    :param coding2: second coding, uncollapsed numpyarray of events
    :param intervals: list of lists where each internal list contains 2 entries indicating start and end time of interval.
    note: intervals are considered as [) i.e. includes start time, but excludes end time.
    :return: array of dictionaries containing various metrics (1 dict per interval)
    """
    results = []
    for interval in intervals:
        t_start, t_end = interval[0], interval[1]
        results.append(get_stats_in_interval(t_start, t_end, coding1, coding2))
    if len(results) == 1:
        results = results[0]
    return results


def compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args):
    """
    compares human coders and machine annotations
    :param human_coding_file:
    :param human_coding_file2:
    :param machine_coding_file:
    :param args:
    :return:
    """
    if args.machine_coding_format == "PrefLookTimestamp":
        parser = parsers.PrefLookTimestampParser(30)
    elif args.machine_coding_format == "princeton":
        parser = parsers.PrincetonParser(30, start_time_file=Path(args.raw_dataset_folder, "start_times_visitA.csv"))
    elif args.machine_coding_format == "compressed":
        parser = parsers.CompressedParser()
    machine, mstart, mend = parser.parse(machine_coding_file)
    trial_times = None
    if args.human_coding_format == "PrefLookTimestamp":
        parser = parsers.PrefLookTimestampParser(30)
    elif args.human_coding_format == "princeton":
        parser = parsers.PrincetonParser(30, start_time_file=Path(args.raw_dataset_folder, "start_times_visitA.csv"))
    elif args.human_coding_format == "lookit":
        parser = parsers.LookitParser(30)
        labels = parser.load_and_sort(human_coding_file)
        trial_times = parser.get_trial_end_times(labels)
        trial_times.insert(0, 0)
        trial_times = [[trial_times[i-1], trial_times[i]] for i, _ in enumerate(trial_times) if i > 0]
    human, start1, end1 = parser.parse(human_coding_file, file_is_fullpath=True)
    human2, start2, end2 = parser.parse(human_coding_file2, file_is_fullpath=True)
    if end1 != end2:
        logging.warning("critical failure: humans don't agree on ending: {}".format(human_coding_file))
        return None
    metrics = {}
    machine_uncol = parser.uncollapse_labels(machine, mstart, mend)
    special_machine = machine_uncol.copy()
    special_machine[special_machine < 0] = 0
    human1_uncol = parser.uncollapse_labels(human, start1, end1)
    human2_uncol = parser.uncollapse_labels(human2, start2, end2)
    # bins = [[x, x+30] for x in range(0, end1, 30)]
    metrics["human1_vs_machine_trials"] = compare_uncollapsed_coding_files(human1_uncol, machine_uncol, trial_times)
    # metrics["human1_vs_machine_100msbins"] = compare_uncollapsed_coding_files(human1_uncol, machine_uncol, bins)
    metrics["human1_vs_machine_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                            machine_uncol,
                                                                            [[0, max(end1, mend)]])
    metrics["human1_vs_smachine_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                             special_machine,
                                                                             [[0, max(end1, mend)]])
    metrics["human1_vs_human2_trials"] = compare_uncollapsed_coding_files(human1_uncol, human2_uncol, trial_times)
    # metrics["human1_vs_human2_100msbins"] = compare_uncollapsed_coding_files(human1_uncol, human2_uncol, bins)
    metrics["human1_vs_human2_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                           human2_uncol,
                                                                           [[0, max(end1, end2)]])
    ICC_looking_time_hvm = calc_ICC(metrics["human1_vs_machine_trials"],
                                    "looking_time_1", "looking_time_2",
                                    len(trial_times))
    ICC_looking_time_hvh = calc_ICC(metrics["human1_vs_human2_trials"],
                                    "looking_time_1", "looking_time_2",
                                    len(trial_times))
    ICC_percent_r_hvm = calc_ICC(metrics["human1_vs_machine_trials"],
                                 "percent_r_1", "percent_r_2",
                                 len(trial_times))
    ICC_percent_r_hvh = calc_ICC(metrics["human1_vs_human2_trials"],
                                 "percent_r_1", "percent_r_2",
                                 len(trial_times))
    metrics["stats"] = {"ICC_LT_hvm": ICC_looking_time_hvm,
                        "ICC_LT_hvh": ICC_looking_time_hvh,
                        "ICC_PR_hvm": ICC_percent_r_hvm,
                        "ICC_PR_hvh": ICC_percent_r_hvh
                        }
    return metrics


def calc_ICC(metrics, dependant_measure1, dependant_measure2, n_trials):
    """
    calculates ICC3 for single fixesd raters (https://pingouin-stats.org/generated/pingouin.intraclass_corr.html)
    :param metrics: dictionary of results upon trials
    :param dependant_measure1: the measure to calculate ICC over by coder1
    :param dependant_measure2: the measure to calculate ICC over by coder2
    :param trial_times: number of trials
    :return: ICC3 metric
    """
    trial_n = np.repeat(np.arange(0, n_trials, 1), 2)
    coders = np.array([[1, 2] for _ in range(n_trials)]).reshape(-1)
    ratings = np.array([[x[dependant_measure1], x[dependant_measure2]] for x in metrics]).reshape(-1)
    df = pd.DataFrame({'trial_n': trial_n,
                       'coders': coders,
                       'ratings': ratings})
    icc = pg.intraclass_corr(data=df, targets='trial_n', raters='coders', ratings='ratings')
    LT_ICC = icc["ICC"][2]
    return LT_ICC


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


def select_frames_from_video(ID, video_folder, start, end):
    """
    selects 9 random frames from a video for display
    :param ID: the video id (no extension)
    :param video_folder: the raw video folder
    :param start: where annotation begins
    :param end: where annotation ends
    :return: an image grid of 9 frames and the corresponding frame numbers
    """
    imgs_np = np.ones((480*3, 640*3, 3))
    for video_file in Path(video_folder).glob("*"):
        if ID in video_file.name:
            imgs = []
            cap = cv2.VideoCapture(str(video_file))
            frame_selections = np.random.choice(np.arange(start, end//2), size=9, replace=False)
            for i in range(start, end//2):  # to avoid end of video
                ret, frame = cap.read()
                if i in frame_selections:
                    imgs.append(frame[..., ::-1])
                    if len(imgs) >= 9:
                        break
            imgs_np = np.array(imgs)
            imgs_np = make_gridview(imgs_np)
    return imgs_np, frame_selections


def sample_luminance(ID, args, start, end, num_samples=10):
    total_luminance = 0
    sampled = 0
    for video_file in args.raw_video_folder.glob("*"):
        if ID in video_file.stem:
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


def generate_frame_by_frame_comparisons(sorted_IDs, all_metrics, args):
    GRAPH_CLASSES = ["left", "right", "away", "invalid"]
    widths = [20, 1, 5]
    heights = [1]
    skip = 10
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    frame_by_frame_path = Path(args.output_folder, "frame_by_frame")
    frame_by_frame_path.mkdir(exist_ok=True, parents=True)
    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        fig, axs = plt.subplots(1, 3, figsize=(24.0, 8.0), gridspec_kw=gs_kw)
        timeline, accuracy, sample_frame = axs  # won't work with single video...
        plt.suptitle('Frame by frame comparisons: {}'.format(target_ID + ".mp4"))
        start1, end1 = all_metrics[target_ID]["human1_vs_human2_session"]["start"],\
                       all_metrics[target_ID]["human1_vs_human2_session"]["end"]
        start2, end2 = all_metrics[target_ID]["human1_vs_machine_session"]["start"], \
                       all_metrics[target_ID]["human1_vs_machine_session"]["end"]
        start = min(start1, start2)
        end = max(end1, end2)
        timeline.set_title("Frames: {} - {}".format(str(start), str(end)))

        times1 = all_metrics[target_ID]["human1_vs_human2_session"]["times_coding2"]
        times2 = all_metrics[target_ID]["human1_vs_human2_session"]["times_coding1"]
        times3 = all_metrics[target_ID]["human1_vs_machine_session"]["times_coding2"]
        times = [times1, times2, times3]
        video_label = ["coder 2", "coder 1", "machine"]
        trial_times = [x["end"] for x in all_metrics[target_ID]["human1_vs_human2_trials"]]
        vlines_handle = timeline.vlines(trial_times, -1, 3, ls='--', color='k', label="trial times")
        for j, vid_label in enumerate(video_label):
            timeline.set_xlabel("Frame #")
            for label in GRAPH_CLASSES:
                timeline.barh(vid_label, skip, left=times[j][label][::skip],
                              height=1, label=label,
                              color=label_to_color(label))
            if j == 0:
                # timeline.legend(loc='upper right')
                accuracy.set_title("Agreement")
        artists = [Patch(facecolor=label_to_color("away"), label="Away"),
                   Patch(facecolor=label_to_color("left"), label="Left"),
                   Patch(facecolor=label_to_color("right"), label="Right"),
                   Patch(facecolor=label_to_color("invalid"), label="Invalid"),
                   vlines_handle]
        timeline.legend(handles=artists, bbox_to_anchor=(0.95, 1.0), loc='upper left')

        inference = ["human1_vs_human2_session", "human1_vs_machine_session"]
        accuracies = [all_metrics[target_ID][entry]['agreement']*100 for entry in inference]
        # colors = [color_gradient[int(acc * 100)].rgb for acc in accuracies]

        accuracy.bar(range(len(inference)), accuracies, color="black")
        accuracy.set_xticks(range(len(inference)))
        accuracy.set_xticklabels(inference, rotation=45, ha="right")
        accuracy.set_ylim([0, 100])
        accuracy.set_ylabel("Agreement")
        # sample_frame_index = min(
        #     [all_metrics[target_ID][ICATCHER]['times_target'][label][0] for label in VALID_CLASSES])
        imgs, times = select_frames_from_video(target_ID, args.raw_video_folder, start, end)
        sample_frame.imshow(imgs)
        sample_frame.set_axis_off()
        # sample_frame_index = (end - start) / 2.0
        # sample_frame.imshow(get_frame_from_video(target_ID, sample_frame_index, args.raw_video_folder))
        longstring = 'Sample frames from video at frames: {}'.format(times)
        formatted_longstring = "\n".join(textwrap.wrap(longstring, 40))
        sample_frame.set_title(formatted_longstring)
        plt.tight_layout()

        plt.savefig(Path(frame_by_frame_path, 'frame_by_frame_{}.png'.format(target_ID + ".mp4")))
        # plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
        plt.cla()
        plt.clf()


def generate_collage_plot2(sorted_IDs, all_metrics, save_path):
    """
    plots one image with various selected stats
    :param sorted_IDs: ids of videos sorted by accuracy score
    :param all_metrics: all metrics per video
    :param save_path: where to save the image
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig = plt.figure(figsize=(10, 12))

    # confusion matrix
    conf_mat_h2h = fig.add_subplot(3, 2, 1)  # three rows, two columns
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_human2_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, -1, keepdims=True)
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Coder 1')
    conf_mat_h2h.set_ylabel('Coder 2')
    conf_mat_h2h.set_title('Confusion Matrix (Coder 1 vs Coder 2)')

    # confusion matrix 2
    conf_mat_h2h = fig.add_subplot(3, 2, 2)
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_machine_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, -1, keepdims=True)
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Coder 1')
    conf_mat_h2h.set_ylabel('Machine')
    conf_mat_h2h.set_title('Confusion Matrix (Coder 1 vs Machine)')

    # confusion matrix 3
    conf_mat_h2h = fig.add_subplot(3, 2, 3)
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_smachine_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, -1, keepdims=True)
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Coder 1')
    conf_mat_h2h.set_ylabel('Machine')
    conf_mat_h2h.set_title(r'Confusion Matrix (Coder 1 vs Machine w "invlid$\leftarrow$away")')

    # LT plot
    lt_scatter = fig.add_subplot(3, 2, 4)
    lt_scatter.plot([0, 1], [0, 1], transform=lt_scatter.transAxes, color="black", label="Ideal trend")
    lt_scatter.set_xlim([0, 100])
    lt_scatter.set_ylim([0, 100])
    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["looking_time_1"]/30 for x in all_metrics[ID]["human1_vs_machine_trials"]]
        y_target += [x["looking_time_2"]/30 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    lt_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.9)
    lt_scatter.set_xlabel("Coder 1")
    lt_scatter.set_ylabel("Machine")
    lt_scatter.set_title("Looking time [s]")
    lt_scatter.legend(loc='upper left')

    # %R plot
    pr_scatter = fig.add_subplot(3, 2, 5)
    pr_scatter.plot([0, 1], [0, 1], transform=pr_scatter.transAxes, color="black", label="Ideal trend")
    pr_scatter.set_xlim([0, 100])
    pr_scatter.set_ylim([0, 100])
    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["percent_r_1"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
        y_target += [x["percent_r_2"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    pr_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.5)
    pr_scatter.set_xlabel("Coder 1")
    pr_scatter.set_ylabel("Machine")
    pr_scatter.set_title("Percent Right")
    pr_scatter.legend(loc='lower center')

    # percent agreement plot
    pa_scatter = fig.add_subplot(3, 2, 6)
    pa_scatter.plot([0, 1], [0, 1], transform=pa_scatter.transAxes, color="black", label="Ideal trend")
    pa_scatter.set_xlim([0, 100])
    pa_scatter.set_ylim([0, 100])
    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["agreement"] * 100 for x in all_metrics[ID]["human1_vs_human2_trials"]]
        y_target += [x["agreement"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    pa_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.3)
    pa_scatter.set_xlabel("Coder 1 vs Coder 2")
    pa_scatter.set_ylabel("Coder 1 vs Machine")
    pa_scatter.set_title("Percent Agreement")
    pa_scatter.legend(loc='upper left')

    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.2, hspace=0.5)
    plt.savefig(Path(save_path, "collage2.png"))
    plt.cla()
    plt.clf()


def generate_collage_plot(sorted_IDs, all_metrics, save_path):
    """
    plots one image with various selected stats
    :param sorted_IDs: ids of videos sorted by accuracy score
    :param all_metrics: all metrics per video
    :param save_path: where to save the image
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig = plt.figure(figsize=(10, 12))

    # accuracies plot
    accuracy_bar = fig.add_subplot(3, 2, (1, 2))  # three rows, two columns
    # accuracy_bar = axs[0, :]
    accuracies_hvh = [all_metrics[ID]["human1_vs_human2_session"]['agreement']*100 for ID in sorted_IDs]
    mean_hvh = np.mean(accuracies_hvh)
    accuracies_hvm = [all_metrics[ID]["human1_vs_machine_session"]['agreement']*100 for ID in sorted_IDs]
    mean_hvm = np.mean(accuracies_hvm)
    labels = sorted_IDs
    width = 0.35  # the width of the bars
    x = np.arange(len(labels))
    accuracy_bar.bar(x - width / 2, accuracies_hvh, width, color=label_to_color("lorange"), label='Human vs Human')
    accuracy_bar.bar(x + width / 2, accuracies_hvm, width, color=label_to_color("mblue"), label='Human vs Machine')
    accuracy_bar.set_ylabel('Percent Agreement')
    accuracy_bar.set_xlabel('Video')
    accuracy_bar.set_title('Percent agreement per video')
    accuracy_bar.set_xticks(x)
    accuracy_bar.axhline(y=mean_hvh, color=label_to_color("lorange"), linestyle='-', label="mean (" + str(mean_hvh)[:4] + ")")
    accuracy_bar.axhline(y=mean_hvm, color=label_to_color("mblue"), linestyle='-', label="mean (" + str(mean_hvm)[:4] + ")")
    accuracy_bar.set_ylim([0, 100])
    accuracy_bar.legend()

    # target valid plot
    transitions_bar = fig.add_subplot(3, 2, 3)  # three rows, two columns
    width = 0.66  # the width of the bars
    x = np.arange(len(sorted_IDs))
    transitions_h1 = [100*all_metrics[ID]["human1_vs_human2_session"]['n_transitions_1'] /
                          all_metrics[ID]["human1_vs_human2_session"]['valid_frames_1'] for ID in sorted_IDs]
    transitions_h2 = [100*all_metrics[ID]["human1_vs_human2_session"]['n_transitions_2'] /
                          all_metrics[ID]["human1_vs_human2_session"]['valid_frames_2'] for ID in sorted_IDs]
    transitions_m = [100*all_metrics[ID]["human1_vs_machine_session"]['n_transitions_2'] /
                          all_metrics[ID]["human1_vs_machine_session"]['valid_frames_2'] for ID in sorted_IDs]

    transitions_bar.bar(x - width / 3, transitions_h1, width=(width / 3) - 0.1, label="Human 1", color=label_to_color("lorange"))
    transitions_bar.bar(x, transitions_h2, width=(width / 3) - 0.1, label="Human 2", color=label_to_color("lgreen"))
    transitions_bar.bar(x + width / 3, transitions_m, width=(width / 3) - 0.1, label="Machine", color=label_to_color("mblue"))
    transitions_bar.set_xticks(x)
    transitions_bar.set_title('# Transitions per 100 frames')
    transitions_bar.legend()
    transitions_bar.set_ylabel('# Transitions per 100 frames')
    transitions_bar.set_xlabel('Video')

    # Looking time plot
    on_away_scatter = fig.add_subplot(3, 2, 4)  # three rows, two columns
    # on_away_scatter = axs[1, 1]
    on_away_scatter.plot([0, 1], [0, 1], transform=on_away_scatter.transAxes, color="black", label="Ideal trend")
    on_away_scatter.set_xlim([0, 600])
    on_away_scatter.set_ylim([0, 600])
    x_target_away_hvh = [all_metrics[ID]["human1_vs_human2_session"]['looking_time_1']/30 for ID in sorted_IDs]
    y_target_away_hvh = [all_metrics[ID]["human1_vs_human2_session"]['looking_time_2']/30 for ID in sorted_IDs]
    x_target_away_hvm = [all_metrics[ID]["human1_vs_machine_session"]['looking_time_1']/30 for ID in sorted_IDs]
    y_target_away_hvm = [all_metrics[ID]["human1_vs_machine_session"]['looking_time_2']/30 for ID in sorted_IDs]
    on_away_scatter.scatter(x_target_away_hvh, y_target_away_hvh, color=label_to_color("lorange"), label='Human vs Human')
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away_hvh[i], y_target_away_hvh[i]))
    on_away_scatter.scatter(x_target_away_hvm, y_target_away_hvm, color=label_to_color("mblue"), label='Human vs Machine')
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away_hvm[i], y_target_away_hvm[i]))
    on_away_scatter.set_xlabel("Human 1")
    on_away_scatter.set_ylabel("Human 2 or Machine")
    on_away_scatter.set_title("Looking time [s]")
    on_away_scatter.legend(loc='upper right')

    # label distribution plot
    # label_scatter = fig.add_subplot(3, 2, 5)  # three rows, two columns
    # # label_scatter = axs[2, 0]
    # label_scatter.plot([0, 1], [0, 1], transform=label_scatter.transAxes, color="black", label="Ideal trend")
    # label_scatter.set_xlim([0, 1])
    # label_scatter.set_ylim([0, 1])
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='^')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='o')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    #
    # label_scatter.set_xlabel('Human 1 label proportion')
    # label_scatter.set_ylabel('Human 2 / Machine labels proportion')
    # label_scatter.set_title('labels distribution')
    # label_scatter.legend()  # loc='upper center'

    # label distribution bar plot
    label_bar = fig.add_subplot(3, 2, (5, 6))  # three rows, two columns
    # label_bar = axs[2, 1]
    ticks = range(len(sorted_IDs))
    bottoms_h1 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_h2 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_m = np.zeros(shape=(len(sorted_IDs)))
    width = 0.66
    patterns = [None, "O", "*"]
    for i, label in enumerate(sorted(classes.keys())):
        label_counts_h1 = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_human2_session"]['label_count_1'] for ID in sorted_IDs]]
        label_counts_h2 = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_human2_session"]['label_count_2'] for ID in sorted_IDs]]
        label_counts_m = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_machine_session"]['label_count_2'] for ID in sorted_IDs]]

        label_bar.bar(x - width/3, label_counts_h1, bottom=bottoms_h1, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[0], linewidth=0)
        label_bar.bar(x, label_counts_h2, bottom=bottoms_h2, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[1], linewidth=0)
        label_bar.bar(x + width/3, label_counts_m, bottom=bottoms_m, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[2], linewidth=0)
        if i == 0:
            artists = [Patch(facecolor=label_to_color("away"), label="Away"),
                       Patch(facecolor=label_to_color("left"), label="Left"),
                       Patch(facecolor=label_to_color("right"), label="Right"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[0], label="Human 1"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[1], label="Human 2"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[2], label="Machine")]
            label_bar.legend(handles=artists, bbox_to_anchor=(0.95, 1.0), loc='upper left')
        bottoms_h1 += label_counts_h1
        bottoms_h2 += label_counts_h2
        bottoms_m += label_counts_m
    label_bar.xaxis.set_major_locator(MultipleLocator(1))
    label_bar.set_xticks(ticks)
    label_bar.set_title('Proportion of looking left, right, and away per video')
    label_bar.set_ylabel('Proportion')
    label_bar.set_xlabel('Video')

    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.2, hspace=0.5)
    plt.savefig(Path(save_path, "collage.png"))
    plt.cla()
    plt.clf()


def plot_inference_accuracy_vs_human_agreement(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    plt.scatter([all_metrics[id]["human1_vs_human2_session"]["agreement"] for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine_session"]["agreement"] for id in sorted_IDs])
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel("Human accuracy")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy vs human accuracy for all doubly coded videos")
    plt.savefig(Path(args.output_folder, 'iCatcher_acc_vs_human_acc.png'))
    plt.cla()
    plt.clf()


def plot_luminance_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    plt.scatter([sample_luminance(id, args,
                                  all_metrics[id]["human1_vs_machine_session"]['start'],
                                  all_metrics[id]["human1_vs_machine_session"]['end']) for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine_session"]["agreement"] for id in sorted_IDs])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel("Luminance")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus mean video luminance for all doubly coded videos")
    plt.savefig(Path(args.output_folder, 'iCatcher_lum_vs_acc.png'))
    plt.cla()
    plt.clf()


def get_face_pixel_density(id, faces_folder):
    """
    given a video id, calculates the average face area in pixels using pre-processed crops
    :param ids: video id
    :param faces_folder: the folder containing all crops and their meta data as created by "preprocess.py"
    :return:
    """
    face_areas = []
    file = Path(faces_folder, id, 'face_labels_fc.npy')
    if file.is_file():
        face_labels = np.load(file)
        for i, face_id in enumerate(face_labels):
            if face_id >= 0:
                box_file = Path(faces_folder, id, "box", "{:05d}_{:01d}.npy".format(i, face_id))
                '{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy.format()'
                box = np.load(box_file, allow_pickle=True).item()
                face_area = (box['face_box'][1] - box['face_box'][0]) * (box['face_box'][3] - box['face_box'][2])
                face_areas.append(face_area)
        return np.mean(face_areas)
    else:
        return None


def get_face_location_std(id, faces_folder):
    """
    given a video id, calculates the standard deviation of the face location in that video
    :param ids: video id
    :param faces_folder: the folder containing all crops and their meta data as created by "preprocess.py"
    :return:
    """
    movements = []
    file = Path(faces_folder, id, 'face_labels_fc.npy')
    if file.is_file():
        face_labels = np.load(file)
        for i, face_id in enumerate(face_labels):
            if face_id >= 0:
                box_file = Path(faces_folder, id, "box", "{:05d}_{:01d}.npy".format(i, face_id))
                '{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy.format()'
                box = np.load(box_file, allow_pickle=True).item()
                face_loc = np.array([box['face_box'][1] - box['face_box'][0], box['face_box'][3] - box['face_box'][2]])
                movements.append(face_loc)
        movements = np.array(movements)
        stds = np.mean(np.std(movements, axis=0))
        return stds
    else:
        return None


def plot_face_pixel_density_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    densities = [all_metrics[x]["stats"]["avg_face_pixel_density"] for x in sorted_IDs]
    plt.scatter(densities, [all_metrics[id]["human1_vs_machine_session"]["agreement"] for id in sorted_IDs])
    plt.xlabel("Face pixel denisty")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus average face pixel density per video")
    plt.savefig(Path(args.output_folder, 'iCatcher_face_density_acc.png'))
    plt.cla()
    plt.clf()


def plot_face_location_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    stds = [all_metrics[x]["stats"]["avg_face_loc_std"] for x in sorted_IDs]
    plt.scatter(stds, [all_metrics[id]["human1_vs_machine_session"]["agreement"] for id in sorted_IDs])
    plt.xlabel("Face location std in pixels")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus face location pixel std")
    plt.savefig(Path(args.output_folder, 'iCatcher_face_location_std_acc.png'))
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

        # Get a list of all machine annotation files
        for file in Path(args.human_codings_folder).glob("*"):
            human_annotation.append(file.stem)
            human_ext = file.suffix
        for file in Path(args.human2_codings_folder).glob("*"):
            human_annotation2.append(file.stem)
            human2_ext = file.suffix
        for file in Path(args.machine_codings_folder).glob("*"):
            machine_annotation.append(file.stem)
            machine_ext = file.suffix

        coding_intersect = set(human_annotation2).intersection(set(human_annotation))
        coding_intersect = coding_intersect.intersection(set(machine_annotation))
        # sort the file paths alphabetically to pair them up
        coding_intersect = sorted(list(coding_intersect))
        assert len(coding_intersect) > 0
        all_metrics = {}
        for i, code_file in enumerate(coding_intersect):
            logging.info("computing stats: {} / {}".format(i, len(coding_intersect) - 1))
            human_coding_file = Path(args.human_codings_folder, code_file + human_ext)
            human_coding_file2 = Path(args.human2_codings_folder, code_file + human2_ext)
            machine_coding_file = Path(args.machine_codings_folder, code_file + machine_ext)
            key = human_coding_file.stem
            res = compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args)
            if res:
                all_metrics[key] = res
                # other stats
                all_metrics[key]["stats"]["avg_face_pixel_density"] = get_face_pixel_density(key, args.faces_folder)
                all_metrics[key]["stats"]["avg_face_loc_std"] = get_face_location_std(key, args.faces_folder)
        # Store in disk for faster access next time:
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


def make_gridview(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def sandbox(metrics):
    for key in metrics.keys():
        acc_human = metrics[key]["human1_vs_human2"]["accuracy"]
        if acc_human >= 100.0:
            print("{}, human accuracy: {}".format(key.stem + ".mp4", acc_human))
        acc_machine = metrics[key]["human1_vs_machine"]["accuracy"]
        if acc_machine <= 60.0:
            print("{}, machine accuracy: {}".format(key.stem + ".mp4", acc_machine))


if __name__ == "__main__":
    args = parse_arguments_for_visualizations()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    # preprocess.create_annotation_split(Path(args.output_folder), Path(args.raw_dataset_folder))
    all_metrics = create_cache_metrics(args, force_create=False)
    # sort by accuracy
    sorted_ids = sorted(list(all_metrics.keys()),
                        key=lambda x: all_metrics[x]["human1_vs_machine_session"]["agreement"])
    generate_collage_plot(sorted_ids, all_metrics, args.output_folder)
    generate_collage_plot2(sorted_ids, all_metrics, args.output_folder)
    generate_frame_by_frame_comparisons(sorted_ids, all_metrics, args)
    plot_face_pixel_density_vs_accuracy(sorted_ids, all_metrics, args)
    plot_face_location_vs_accuracy(sorted_ids, all_metrics, args)
    plot_inference_accuracy_vs_human_agreement(sorted_ids, all_metrics, args)
    plot_luminance_vs_accuracy(sorted_ids, all_metrics, args)


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
