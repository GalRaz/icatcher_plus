import csv
import os
import cv2
import numpy as np
from tqdm import tqdm
import config

"""
Script for calculating and saving video metadata as a csv
"""


def calculate_luminance(cap, start, end, num_samples=100):
    """
    Calculates the average luminance sampled over num_samples frames between start and end
    """
    total_luminance = 0
    sampled = 0
    sample_every = int((end - start) / num_samples)
    frame_no = start
    ret, frame = cap.read()
    while ret:
        b, g, r = cv2.split(frame)
        total_luminance += (0.2126 * np.sum(r) + 0.7152 * np.sum(g) + 0.0722 * np.sum(b)) / (len(b) * len(b[0]))
        sampled += 1
        frame_no += sample_every
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
    return total_luminance // sampled

def lucas_kanade_method(video_path):
    # Read the video
    cap = cv2.VideoCapture(str(video_path))
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a,b,c,d=int(a),int(b),int(c),int(d)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # Display the demo
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

def calculate_motion_energy(video_filename, start, end):
    """
    Calculates the average motion energy over the coding active region
    """
    lucas_kanade_method(config.video_folder/video_filename)


def save_csv(csv_headers, columns, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(csv_headers)
        for video_id in columns["video id"]:
            row = [video_id]
            for column_header in columns.keys():
                if not column_header == "video id":
                    try:
                        row.append(columns[column_header][video_id])
                    except KeyError:
                        row.append("")
            csv_writer.writerow(row)


def get_coding_active_range(label_filename):
    labels = np.genfromtxt(open(config.label_folder / label_filename, "rb"), dtype='str', delimiter="\t", skip_header=1)
    frames = set()
    for label in labels:
        frame_index = label.split(",")[0]
        try:
            frames.add(int(frame_index))
        except ValueError:
            pass
    return min(frames), max(frames)


def calculate_and_save_all(ordered_video_ids, id_to_video_filename, id_to_coding_first_paths,
                           id_to_coding_second_paths):
    csv_headers = ["video id", "video filename", "coding first filename", "coding second filename", "luminance",
                   "motion energy", "resolution", "fps", "frame count", "coding start (frame)", "coding end (frame)"]
    columns = {header: {} for header in csv_headers}
    columns[csv_headers[0]] = ordered_video_ids
    columns[csv_headers[1]] = id_to_video_filename
    columns[csv_headers[2]] = id_to_coding_first_paths
    columns[csv_headers[3]] = id_to_coding_second_paths

    for video_id in tqdm(ordered_video_ids):
        video_filename = id_to_video_filename[video_id]
        cap = cv2.VideoCapture(str(config.video_folder / video_filename))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f'{width}X{height}'
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        columns[csv_headers[6]][video_id] = resolution
        columns[csv_headers[7]][video_id] = fps
        if video_id in id_to_coding_first_paths:
            start_ms, end_ms = get_coding_active_range(id_to_coding_first_paths[video_id])
            start_frame = start_ms / 1000 * fps
            end_frame = end_ms / 1000 * fps
            luminance = calculate_luminance(cap, start_frame, end_frame)

            columns[csv_headers[4]][video_id] = luminance
            columns[csv_headers[5]][video_id] = calculate_motion_energy(video_filename, start_frame, end_frame)
            columns[csv_headers[8]][video_id] = total_num_frames
            columns[csv_headers[9]][video_id] = start_frame
            columns[csv_headers[10]][video_id] = end_frame

    save_csv(csv_headers, columns, csv_path="/home/jupyter/metadata.csv")


if __name__ == "__main__":
    ordered_video_ids = []
    id_to_video_filename = {}
    id_to_coding_first_paths = {}
    id_to_coding_second_paths = {}

    coding_first = set([f for s, d, f in os.walk(config.label_folder)][0])
    coding_second = set([f for s, d, f in os.walk(config.label2_folder)][0])

    for _, _, filenames in os.walk(config.video_folder):
        for filename in filenames:
            video_id = str(filename).replace('cfddb63f-12e9-4e62-abd1-47534d6c4dd2_', '').replace('.mp4', '')
            ordered_video_ids.append(video_id)
            id_to_video_filename[video_id] = filename

    for _, _, filenames in os.walk(config.label_folder):
        for filename in filenames:
            for video_id in ordered_video_ids:
                if video_id in filename:
                    id_to_coding_first_paths[video_id] = filename
                    break

    for _, _, filenames in os.walk(config.label2_folder):
        for filename in filenames:
            for video_id in ordered_video_ids:
                if video_id in filename:
                    id_to_coding_second_paths[video_id] = filename
                    break
    calculate_and_save_all(ordered_video_ids, id_to_video_filename, id_to_coding_first_paths, id_to_coding_second_paths)
