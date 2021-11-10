import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import numpy as np


class BaseParser:
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2}

    def parse(self, file):
        """
        returns a list of lists. each list contains the frame number, valid_flag, class
        where:
        frame number is zero indexed
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.

        list should only contain frames that have changes in class (compared to previous frame)
        i.e. if the video is labled ["away","away","away","right","right"]
        then only frame 0 and frame 3 will appear on the output list.

        :param file: the label file to parse.
        :return: None if failed, else: list of lists as described above
        """
        raise NotImplementedError


class TrivialParser(BaseParser):
    """
    A trivial parser that labels all video as "left" if input "file" is not None
    """
    def __init__(self):
        super().__init__()

    def parse(self, file):
        if file:
            return [[0, 1, "left"]]
        else:
            return None


class PrefLookTimestampParser:
    """
    a parser that can parse PrefLookTimestamp as described here:
    https://osf.io/3n97m/
    """
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2}

    def parse(self, file, fps=1000):
        """
        returns a list of lists. each list contains the frame number, valid_flag, class
        where:
        frame number is zero indexed
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.
        list should only contain frames that have changes in class (compared to previous frame)
        i.e. if the video is labled ["away","away","away","right","right"]
        then only frame 0 and frame 3 will appear on the output list.
        :param file: the label file to parse.
        :return: None if failed, else: list of lists as described above
        """

        labels = np.genfromtxt(open(file, "rb"), dtype='str', delimiter=",", skip_header=3)
        output = []
        num_frames = int(labels[-1, 1])
        last_label = "left" #arbitrary, doesn't matter until valid flag gets changed to 1
        valid_flag = 0
        start, end = 0, 0
        time_to_frame = lambda time: int(time * fps / 1000.0)

        for entry in range(labels.shape[0]):
            
            frame = time_to_frame(int(labels[entry, 0]))
            label = labels[entry, 2]
                
            class_flag = label#self.classes[label]
            valid_flag = 1 if class_flag in self.classes else 0
            if label == "codingactive":
                start, end = time_to_frame(int(labels[entry, 0])), time_to_frame(int(labels[entry, 1]))
                frame = int(labels[entry, 1])
            frame_label = [frame, valid_flag, class_flag]
            
        
            # frame_label = [frame, valid_flag, class_flag]
            output.append(frame_label)
            # print(frame_label)
        output.sort(key=lambda x: x[0])
        if end == 0:
            end = int(output[-1][0])
        return output, start, end


class PrincetonParser(BaseParser):
    """
    A parser that can parse vcx files that are used in princeton laboratories
    """
    def __init__(self, ext, labels_folder):
        super().__init__()
        self.ext = ext
        self.labels_folder = Path(labels_folder)

    def parse(self, file, fps=30):
        label_path = Path(self.labels_folder, file + self.ext)
        if not label_path.is_file():
            logging.warning("For the file: " + str(file) + " no matching xml was found.")
            return None
        else:
            return self.xml_parse(label_path, fps, True)

    def xml_parse(self, input_file, fps, encode=False):
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
            response_hours = [int(x[1][6]) for x in sorted_responses]
            if not response_hours.count(response_hours[0]) == len(response_hours):
                logging.warning("how can the video span more than 1 hour?")
                raise ValueError
            for response in sorted_responses:
                # do not count hours. assumption here is no video spans more than an hour.
                # if yes, you can use + int(response[1][6]) * 60 * 60 * fps
                frame_number = int(response[1][4]) + int(response[1][10]) * fps + int(response[1][8]) * 60 * fps
                encoded_responses.append([frame_number, response[1][14], response[1][16]])
            sorted_responses = encoded_responses
        # replace offs with aways, they are equivalent
        for i, item in enumerate(sorted_responses):
            if item[2] == 'off':
                item[2] = 'away'
                sorted_responses[i] = item
        return sorted_responses
