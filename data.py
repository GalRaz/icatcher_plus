import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import copy
from pathlib import Path
import logging
import csv


class DataTransforms:
    def __init__(self, img_size):
        self.transformations = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
}


class LookItDataset:
    def __init__(self, args):
        self.args = args
        self.transforms = DataTransforms(args.image_size).transformations
        self.img_processor = self.transforms[args.phase]
        if self.args.filter_validation and self.args.phase == "val":
            self.file_filter = self.parse_filter_file()
        else:
            self.file_filter = None
        self.paths = self.collect_paths("face_labels_fc")  # change to "face_labels" if face classifier wasn't trained

    def __len__(self):
        return len(self.paths)

    def check_all_same(self, seg):
        """
        checks if all labels are the same
        :param seg:
        :return:
        """
        for i in range(1, seg.shape[0]):
            if seg[i] != seg[i - 1]:
                return False
        return True

    def parse_filter_file(self):
        validation_videos = []
        disjoint = "FALSE" if self.args.use_disjoint else "TRUE"
        with open(self.args.filter_validation, newline='') as csvfile:
            my_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in my_reader:
                if row[0] == "test" and row[-1] == disjoint:
                    validation_videos.append(Path(row[1]).stem)
        return validation_videos

    def collect_paths(self, face_label_name):
        """
        process dataset into tuples of frames
        :param face_label_name: file with face labels
        :return:
        """
        all_names_path = Path(self.args.dataset_folder, "coding_first")
        test_names_path = Path(self.args.dataset_folder, "coding_second")
        dataset_folder_path = Path(self.args.dataset_folder, "faces")
        all_names = [f.stem for f in all_names_path.glob('*.txt')]
        test_names = [f.stem for f in test_names_path.glob('*.txt')]
        my_list = []
        logging.info("Collecting paths for dataloader...")
        video_counter = 0
        for name in all_names:
            if self.args.phase == "val":
                if name not in test_names:
                    continue
                if self.file_filter:
                    if name not in self.file_filter:
                        continue
            elif self.args.phase == "train":
                if name in test_names:
                    continue
            else:
                raise NotImplementedError
            gaze_labels = np.load(str(Path.joinpath(dataset_folder_path, name, f'gaze_labels.npy')))
            face_labels = np.load(str(Path.joinpath(dataset_folder_path, name, f'{face_label_name}.npy')))
            for frame_number in range(gaze_labels.shape[0]):
                gaze_label_seg = gaze_labels[frame_number:frame_number + self.args.frames_per_datapoint]
                face_label_seg = face_labels[frame_number:frame_number + self.args.frames_per_datapoint]
                if len(gaze_label_seg) != self.args.frames_per_datapoint:
                    break
                if sum(face_label_seg < 0):
                    continue
                if not self.args.eliminate_transitions or self.check_all_same(gaze_label_seg):
                    class_seg = gaze_label_seg[self.args.frames_per_datapoint // 2]
                    img_files_seg = []
                    box_files_seg = []
                    for i in range(self.args.frames_per_datapoint):
                        img_files_seg.append(f'{name}/img/{frame_number + i:05d}_{face_label_seg[i]:01d}.png')
                        box_files_seg.append(f'{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy')
                    img_files_seg = img_files_seg[::self.args.frames_stride_size]
                    box_files_seg = box_files_seg[::self.args.frames_stride_size]
                    my_list.append((img_files_seg, box_files_seg, class_seg))
            if not my_list:
                logging.info("The video {} has no annotations".format(name))
                continue
            video_counter += 1
        logging.info("Used {} videos, for a total of {} datapoints".format(video_counter, len(my_list)))
        return my_list

    def __getitem__(self, index):
        img_files_seg, box_files_seg, class_seg = self.paths[index]

        imgs = []
        for img_file in img_files_seg:
            img = Image.open(self.args.dataset_folder / "faces" / img_file).convert('RGB')
            img = self.img_processor(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        boxs = []
        for box_file in box_files_seg:
            box = np.load(self.args.dataset_folder / "faces" / box_file, allow_pickle=True).item()
            box = torch.tensor([box['face_size'], box['face_ver'], box['face_hor'], box['face_height'], box['face_width']])
            boxs.append(box)
        boxs = torch.stack(boxs)
        boxs = boxs.float()
        imgs = imgs.to(self.args.device)
        boxs = boxs.to(self.args.device)
        class_seg = torch.as_tensor(class_seg).to(self.args.device)
        return {
            'imgs': imgs,  # n x 3 x 100 x 100
            'boxs': boxs,  # n x 5
            'label': class_seg
        }


class LookItDatasetStreamed:
    def __init__(self, args):
        self.args = args
        self.transforms = DataTransforms(args.image_size).transformations
        self.img_processor = self.transforms[args.phase]
        self.streams = self.collect_streams("face_labels_fc")  # change to "face_labels" if face classifier wasn't trained

    def __len__(self):
        return len(self.streams[0])

    def collect_streams(self, face_label_name):
        """
        process dataset into tuples of frames
        :param face_label_name: file with face labels
        :return:
        """
        all_names_path = Path(self.args.dataset_folder, "coding_first")
        test_names_path = Path(self.args.dataset_folder, "coding_second")
        dataset_folder_path = Path(self.args.dataset_folder, "faces")
        all_names = [f.stem for f in all_names_path.glob('*.txt')]
        test_names = [f.stem for f in test_names_path.glob('*.txt')]
        my_list = []
        for name in all_names:
            if self.args.phase == "val":
                if name not in test_names:
                    continue
            elif self.args.phase == "train":
                if name in test_names:
                    continue
            else:
                raise NotImplementedError
            gaze_labels = np.load(str(Path.joinpath(dataset_folder_path, name, f'gaze_labels.npy')))
            face_labels = np.load(str(Path.joinpath(dataset_folder_path, name, f'{face_label_name}.npy')))
            for frame_number in range(gaze_labels.shape[0]):
                gaze_label_seg = gaze_labels[frame_number:frame_number + self.args.frames_per_datapoint]
                face_label_seg = face_labels[frame_number:frame_number + self.args.frames_per_datapoint]
                if len(gaze_label_seg) != self.args.frames_per_datapoint:
                    diff = (frame_number + self.args.frames_per_datapoint) - len(gaze_labels)
                    assert diff == 1  # verify we are at the end of the video
                    for i in range(self.args.frames_per_datapoint):  # add blank frames with no annotation
                        my_list.append(("blank", "blank", -2))
                    break
                class_seg = gaze_label_seg[self.args.frames_per_datapoint // 2]
                img_files_seg = []
                box_files_seg = []
                for i in range(self.args.frames_per_datapoint):
                    img_files_seg.append(f'{name}/img/{frame_number + i:05d}_{face_label_seg[i]:01d}.png')
                    box_files_seg.append(f'{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy')
                img_files_seg = img_files_seg[::self.args.frames_stride_size]
                box_files_seg = box_files_seg[::self.args.frames_stride_size]
                my_list.append((img_files_seg, box_files_seg, class_seg))
            if not my_list:
                logging.info("The video {} has no annotations".format(name))
            else:
                break
        splits_size = len(my_list) // self.args.batch_size
        streams = []
        for i in range(self.args.batch_size):
            streams.append(my_list[i*splits_size:(i+1)*splits_size])
        return streams

    def __getitem__(self, index):
        img_files_seg, box_files_seg, class_seg = self.streams[index]

        imgs = []
        for img_file in img_files_seg:
            if img_file != "blank":
                img = Image.open(self.args.dataset_folder / "faces" / img_file).convert('RGB')
                img = self.img_processor(img)
            else:
                img = torch.zeros(100, 100).float()
            imgs.append(img)
        imgs = torch.stack(imgs)

        boxs = []
        for box_file in box_files_seg:
            if box_file != "blank":
                box = np.load(self.args.dataset_folder / "faces" / box_file, allow_pickle=True).item()
                box = torch.tensor(
                    [box['face_size'], box['face_ver'], box['face_hor'], box['face_height'], box['face_width']])
            else:
                box = torch.zeros(5)
            boxs.append(box)
        boxs = torch.stack(boxs)
        boxs = boxs.float()
        imgs = imgs.to(self.args.device)
        boxs = boxs.to(self.args.device)
        class_seg = torch.as_tensor(class_seg).to(self.args.device)
        return {
            'imgs': imgs,  # n x 3 x 100 x 100
            'boxs': boxs,  # n x 5
            'label': class_seg
        }


class MyDataLoader:
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        shuffle = (self.opt.phase == "train")
        self.dataset = LookItDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=shuffle,
            num_workers=int(opt.num_threads)
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data