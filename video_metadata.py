import csv
import os

import cv2
import numpy as np

import config

"""
Script for calculating and saving video metadata as a csv
"""


def get_video(video_filename):
    return cv2.VideoCapture(str(config.video_folder / video_filename))


def calculate_video_fps(video_filename):
    return get_video(video_filename).get(cv2.CAP_PROP_FPS)


def calculate_video_resolution(video_filename):
    cap = get_video(video_filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f'{width}X{height}'

def calculate_video_num_frames(video_filename):
    cap = get_video(video_filename)
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)
def calculate_luminance(video_filename):
    total_luminance = 0
    sampled = 0
    cap = get_video(video_filename)
    sample_every = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 10)
    # print(sample_every)
    frame_no = 0
    ret, frame = cap.read()
    while ret:
        b, g, r = cv2.split(frame)
        total_luminance += (0.2126 * np.sum(r) + 0.7152 * np.sum(g) + 0.0722 * np.sum(b))/(len(b)*len(b[0]))
        sampled += 1
        frame_no += sample_every
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
    return total_luminance // sampled


def calculate_motion_energy(video_filename):
    return 'not calculated yet'


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
                    except:
                        row.append("missing")
            csv_writer.writerow(row)


def calculate_and_save_all(ordered_video_ids, id_to_video_filename, id_to_coding_first_paths,
                           id_to_coding_second_paths):
    csv_headers = ["video id", "video filename", "coding first filename", "coding second filename", "luminosity",
                   "motion energy", "resolution", "fps", "frame count"]
    columns = {}
    columns[csv_headers[0]] = ordered_video_ids
    columns[csv_headers[1]] = id_to_video_filename
    columns[csv_headers[2]] = id_to_coding_first_paths
    columns[csv_headers[3]] = id_to_coding_second_paths
    columns[csv_headers[4]] = {video_id: calculate_luminance(video_filename) for video_id, video_filename in
                               id_to_video_filename.items()}
    columns[csv_headers[5]] = {video_id: calculate_motion_energy(video_filename) for video_id, video_filename in
                               id_to_video_filename.items()}
    columns[csv_headers[6]] = {video_id: calculate_video_resolution(video_filename) for video_id, video_filename in
                               id_to_video_filename.items()}
    columns[csv_headers[7]] = {video_id: calculate_video_fps(video_filename) for video_id, video_filename in
                               id_to_video_filename.items()}
    columns[csv_headers[8]] = {video_id: calculate_video_num_frames(video_filename) for video_id, video_filename in
                               id_to_video_filename.items()}

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
            if video_id in filename:
                id_to_coding_second_paths[video_id] = filename
                break
    calculate_and_save_all(ordered_video_ids, id_to_video_filename, id_to_coding_first_paths, id_to_coding_second_paths)
