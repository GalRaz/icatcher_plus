import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import numpy as np
import csv


class BaseParser:
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2}

    def parse(self, file):
        """
        returns a list of lists. each list contains the frame number (or timestamps), valid_flag, class
        where:
        frame number is zero indexed (or if timestamp, starts from 0.0)
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.

        list should only contain frames that have changes in class (compared to previous frame)
        i.e. if the video is labled ["away","away","away","right","right"]
        then only frame 0 and frame 3 will appear on the output list.

        :param file: the label file to parse.
        :return: None if failed, else: list of lists as described above, the frame which codings starts, and frame at which it ends
        """
        raise NotImplementedError


class TrivialParser(BaseParser):
    """
    A trivial toy parser that labels all video as "left" if input "file" is not None
    """
    def __init__(self):
        super().__init__()

    def parse(self, file):
        if file:
            return [[0, 1, "left"]]
        else:
            return None


class PrefLookTimestampParser(BaseParser):
    """
    a parser that can parse PrefLookTimestamp as described here:
    https://osf.io/3n97m/
    """
    def __init__(self, fps, labels_folder=None, ext=None, return_time_stamps=False):
        super().__init__()
        self.fps = fps
        self.return_time_stamps = return_time_stamps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)

    def parse(self, file, file_is_fullpath=False):
        """
        Parses a label file from the lookit dataset, see base class for output format
        :param file: the file to parse
        :param file_is_fullpath: if true, the file represents a full path and extension,
         else uses the initial values provided.
        :return:
        """
        codingactive_counter = 0
        classes = {"away": 0, "left": 1, "right": 2}
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        labels = np.genfromtxt(open(label_path, "rb"), dtype='str', delimiter=",", skip_header=3)
        output = []
        start, end = 0, 0
        for entry in range(labels.shape[0]):
            if self.return_time_stamps:
                frame = int(labels[entry, 0])
                dur = int(labels[entry, 1])
            else:
                frame = int(int(labels[entry, 0]) * self.fps / 1000)
                dur = int(int(labels[entry, 1]) * self.fps / 1000)
            class_name = labels[entry, 2]
            valid_flag = 1 if class_name in classes else 0
            if class_name == "codingactive":  # indicates the period of video when coding was actually performed
                codingactive_counter += 1
                start, end = frame, dur
                frame = dur  # if codingactive: add another annotation signaling invalid frames from now on
            frame_label = [frame, valid_flag, class_name]
            output.append(frame_label)
        assert codingactive_counter <= 1  # current parser doesnt support multiple coding active periods
        output.sort(key=lambda x: x[0])
        if end == 0:
            end = int(output[-1][0])
        if len(output) > 0:
            return output, start, end
        else:
            return None


class PrincetonParser(BaseParser):
    """
    A parser that can parse vcx files that are used in princeton laboratories
    """
    def __init__(self, fps, ext=None, labels_folder=None, start_time_file=None):
        super().__init__()
        self.fps = fps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)
        self.start_times = None
        if start_time_file:
            self.start_times = self.process_start_times(start_time_file)

    def process_start_times(self, start_time_file):
        start_times = {}
        with open(start_time_file, newline='') as csvfile:
            my_reader = csv.reader(csvfile, delimiter=',')
            next(my_reader, None)  # skip the headers
            for row in my_reader:
                numbers = [int(x) for x in row[1].split(":")]
                time_stamp = numbers[0]*60*60*self.fps + numbers[1]*60*self.fps + numbers[2]*self.fps + numbers[3]
                start_times[Path(row[0]).stem] = time_stamp
        return start_times

    def parse(self, file, file_is_fullpath=False):
        """
        parse a coding file, see base class for output format
        :param file: coding file to parse
        :param file_is_fullpath: if true, the file is a full path with extension, else uses values from initialization
        :return:
        """
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        if not label_path.is_file():
            logging.warning("For the file: " + str(file) + " no matching xml was found.")
            return None
        return self.xml_parse(label_path, True)

    def xml_parse(self, input_file, encode=False):
        tree = ET.parse(input_file)
        root = tree.getroot()
        counter = 0
        frames = {}
        current_frame = ""
        for child in root.iter('*'):
            if child.text is not None:
                if 'Response ' in child.text:
                    current_frame = child.text
                    frames[current_frame] = []
                    counter = 16
                else:
                    if counter > 0:
                        counter -= 1
                        frames[current_frame].append(child.text)
            else:
                if counter > 0:
                    if child.tag == 'true':
                        frames[current_frame].append(1)  # append 1 for true
                    else:
                        frames[current_frame].append(0)  # append 0 for false
        responses = []
        for key, val in frames.items():
            [num] = [int(s) for s in key.split() if s.isdigit()]
            responses.append([num, val])
        sorted_responses = sorted(responses)
        if encode:
            encoded_responses = []
            # response_hours = [int(x[1][6]) for x in sorted_responses]
            # if not response_hours.count(response_hours[0]) == len(response_hours):
            #     logging.warning("response")
            for response in sorted_responses:
                frame_number = int(response[1][4]) +\
                               int(response[1][10]) * self.fps +\
                               int(response[1][8]) * 60 * self.fps +\
                               int(response[1][6]) * 60 * 60 * self.fps
                if self.start_times:
                    start_time = self.start_times[input_file.stem]
                    frame_number -= start_time
                assert frame_number < 60 * 60 * self.fps
                encoded_responses.append([frame_number, response[1][14], response[1][16]])
            sorted_responses = encoded_responses
        # replace offs with aways, they are equivalent
        for i, item in enumerate(sorted_responses):
            if item[2] == 'off':
                item[2] = 'away'
                sorted_responses[i] = item
        start = sorted_responses[0][0]
        end = sorted_responses[-1][0]
        return sorted_responses, start, end
