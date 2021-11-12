import shutil
from pathlib import Path
import cv2
import numpy as np
import time
import logging
import visualize
from PIL import Image
import face_classifier.fc_model
import face_classifier.fc_data
import face_classifier.fc_eval
import torch
from tqdm import tqdm
import options
import parsers


def preprocess_raw_lookit_dataset(args, force_create=False):
    """
    Organizes the raw videos downloaded from the Lookit platform.
    It puts the videos with annotations into raw_videos folder and
    the annotation from the first and second human annotators into coding_first and coding_second folders respectively.
    :param force_create: forces creation of files even if they exist
    :return:
    """
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)

    # TODO: Bugs: 1 file does not have this prefix; 1 file ends with .mov
    prefix = 'cfddb63f-12e9-4e62-abd1-47534d6c4dd2_'
    coding_first = [f.stem[:-5] for f in Path(args.raw_dataset_path / 'coding_first').glob('*.txt')]
    coding_second = [f.stem[:-5] for f in Path(args.raw_dataset_path / 'coding_second').glob('*.txt')]
    videos = [f.stem for f in Path(args.raw_dataset_path / 'videos').glob(prefix+'*.mp4')]

    logging.info('[preprocess_raw] coding_first: {}'.format(len(coding_first)))
    logging.info('[preprocess_raw] coding_second: {}'.format(len(coding_second)))
    logging.info('[preprocess_raw] videos: {}'.format(len(videos)))

    training_set = []
    test_set = []

    for filename in videos:
        if prefix not in filename:
            continue
        label_id = filename[len(prefix):]
        if label_id in coding_first:
            if label_id in coding_second:
                test_set.append(filename)
            else:
                training_set.append(filename)

    logging.info('[preprocess_raw] training set: {} validation set: {}'.format(len(training_set), len(test_set)))

    for filename in training_set:
        if not Path(args.video_folder, (filename[len(prefix):]+'.mp4')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'videos' / (filename+'.mp4'), args.video_folder / (filename[len(prefix):]+'.mp4'))
        if not Path(args.label_folder, (filename[len(prefix):]+'.txt')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'coding_first' / (filename[len(prefix):]+'-evts.txt'), args.label_folder / (filename[len(prefix):]+'.txt'))

    for filename in test_set:
        if not Path(args.video_folder, (filename[len(prefix):] + '.mp4')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'videos' / (filename + '.mp4'), args.video_folder /(filename[len(prefix):]+'.mp4'))
        if not Path(args.label_folder, (filename[len(prefix):] + '.txt')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'coding_first' / (filename[len(prefix):] + '-evts.txt'), args.label_folder / (filename[len(prefix):]+'.txt'))
        if not Path(args.label2_folder, (filename[len(prefix):] + '.txt')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'coding_second' / (filename[len(prefix):] + '-evts.txt'), args.label2_folder / (filename[len(prefix):]+'.txt'))


def preprocess_raw_generic_dataset(args, force_create=False):
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)

    raw_videos_path = Path(args.raw_dataset_path / 'videos')
    raw_coding_first_path = Path(args.raw_dataset_path / 'coding_first')
    raw_coding_second_path = Path(args.raw_dataset_path / 'coding_second')

    videos = [f.stem for f in raw_videos_path.glob("*.mp4")]
    coding_first = [f.stem for f in raw_coding_first_path.glob("*.txt")]
    coding_second = [f.stem for f in raw_coding_second_path.glob("*.txt")]

    logging.info('[preprocess_raw] coding_first: {}'.format(len(coding_first)))
    logging.info('[preprocess_raw] coding_second: {}'.format(len(coding_second)))
    logging.info('[preprocess_raw] videos: {}'.format(len(videos)))

    training_set = set(videos).intersection(set(coding_first))
    test_set = set(videos).intersection(set(coding_first)).intersection(set(coding_second))
    for i, file in enumerate(sorted(list(training_set))):
        if not Path(args.video_folder, (file + '.mp4')).is_file() or force_create:
            shutil.copyfile(raw_videos_path / (file + '.mp4'), args.video_folder / (file + '.mp4'))
        if not Path(args.label_folder, (file + '.txt')).is_file() or force_create:
            shutil.copyfile(raw_coding_first_path / (file + ".txt"), args.label_folder / (file + ".txt"))

    for i, file in enumerate(sorted(list(test_set))):
        if not Path(args.video_folder, (file + '.mp4')).is_file() or force_create:
            shutil.copyfile(raw_videos_path / (file + '.mp4'), args.video_folder / (file + '.mp4'))
        if not Path(args.label_folder, (file + '.txt')).is_file() or force_create:
            shutil.copyfile(raw_coding_first_path / (file + ".txt"), args.label_folder / (file + ".txt"))
        if not Path(args.label2_folder, (file + '.txt')).is_file() or force_create:
            shutil.copyfile(raw_coding_second_path / (file + ".txt"), args.label2_folder / (file + ".txt"))


def preprocess_raw_princeton_dataset(args, force_create=False):
    """
    Organizes the raw videos from marchman/princeton.
    It puts the videos with annotations into raw_videos folder and
    the annotation from the first and second human annotators into coding_first and coding_second folders respectively.
    :param force_create: forces creation of files even if they exist
    :return:
    :param args:
    :param force_create:
    :return:
    """
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)
    videos = [f.stem for f in Path(args.raw_dataset_path / 'Mov').glob("*.mov")]
    coding_first = [f.stem for f in Path(args.raw_dataset_path / 'VCX').glob("*.vcx")]
    coding_second = [f.stem for f in Path(args.raw_dataset_path / 'VCX2').glob("*.vcx")]

    logging.info('[preprocess_raw] coding_first: {}'.format(len(coding_first)))
    logging.info('[preprocess_raw] coding_second: {}'.format(len(coding_second)))
    logging.info('[preprocess_raw] videos: {}'.format(len(videos)))

    training_set = set(videos).intersection(set(coding_first))
    test_set = set(videos).intersection(set(coding_first)).intersection(set(coding_second))
    # val_percent = 20  # 20%
    # train_factor = (100 - val_percent) / 100
    # if i < (int(len(intersection) * train_factor)):
    for i, file in enumerate(sorted(list(training_set))):
        if not Path(args.video_folder, (file + '.mov')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'Mov' / (file + '.mov'), args.video_folder / (file+'.mov'))
        if not Path(args.label_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX' / (file + ".vcx"), args.label_folder / (file + ".vcx"))

    for i, file in enumerate(sorted(list(test_set))):
        if not Path(args.video_folder, (file + '.mov')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'Mov' / (file + '.mov'), args.video_folder / (file+'.mov'))
        if not Path(args.label_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX' / (file + ".vcx"), args.label_folder / (file + ".vcx"))
        if not Path(args.label2_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX2' / (file + ".vcx"), args.label2_folder / (file + ".vcx"))


def parse_lookit_label(file, fps):
    """
    Parses a label file from the lookit dataset
    :param file: the file to parse
    :param fps: fps of the video
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    labels = np.genfromtxt(open(file, "rb"), dtype='str', delimiter=",", skip_header=3)
    output = []
    for entry in range(labels.shape[0]):
        frame = int(int(labels[entry, 0]) * fps / 1000)
        class_name = labels[entry, 2]
        valid_flag = 1 if class_name in classes else 0
        frame_label = [frame, valid_flag, class_name]
        output.append(frame_label)
    output.sort(key=lambda x: x[0])
    if len(output):
        return output
    else:
        return None


def detect_face_opencv_dnn(net, frame, conf_threshold):
    """
    Uses a pretrained face detection model to generate facial bounding boxes,
    with the format [x, y, width, height] where [x, y] is the lower left coord
    :param net:
    :param frame:
    :param conf_threshold:
    :return:
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)  # left side of box
            y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)  # top side of box
            if x1 >= frameWidth or y1 >= frameHeight:  # if they are larger than image size, bbox is invalid
                continue
            x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)  # either right side of box or frame width
            y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)  # either the bottom side of box of frame height
            bboxes.append([x1, y1, x2-x1, y2-y1])  # (left, top, width, height)
    return bboxes


def process_dataset_lowest_face(args, gaze_labels_only=False, force_create=False):
    """
    process the dataset using the "lowest" face mechanism
    creates:
        face crops (all recognized face crops that abide a threshold value 0.7)
        boxes (the crop parameters - area / location etc)
        gaze_labels (the label of the crop as annotated by first human)
        face_labels (index of the face that was selected using lowest face mechanism)
    :param gaze_labels_only: only creates gaze labels.
    :param force_create: forces creation of files even if they exist
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    video_list = sorted(list(args.video_folder.glob("*")))
    net = cv2.dnn.readNetFromCaffe(str(args.config_file), str(args.face_model_file))
    for video_file in video_list:
        st_time = time.time()
        logging.info("[process_lkt_legacy] Proccessing %s" % video_file.name)
        cur_video_folder = Path.joinpath(args.faces_folder / video_file.stem)
        cur_video_folder.mkdir(parents=True, exist_ok=True)
        img_folder = Path.joinpath(args.faces_folder, video_file.stem, 'img')
        img_folder.mkdir(parents=True, exist_ok=True)
        box_folder = Path.joinpath(args.faces_folder, video_file.stem, 'box')
        box_folder.mkdir(parents=True, exist_ok=True)

        frame_counter = 0
        no_face_counter = 0
        no_annotation_counter = 0
        valid_counter = 0
        gaze_labels = []
        face_labels = []

        cap = cv2.VideoCapture(str(video_file))
        if args.raw_dataset_type == "lookit" or args.raw_dataset_type == "generic":
            responses = parse_lookit_label(args.label_folder / (video_file.stem + '.txt'), cap.get(cv2.CAP_PROP_FPS))
        elif args.raw_dataset_type == "princeton":
            parser = parsers.PrincetonParser(".vcx", args.label_folder)
            responses = parser.parse(video_file.stem)
        else:
            raise NotImplementedError
        ret_val, frame = cap.read()

        while ret_val:
            if responses:
                if frame_counter >= responses[0][0]:  # skip until reaching first annotated frame
                    # find closest (previous) response this frame belongs to
                    q = [index for index, val in enumerate(responses) if frame_counter >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1] != 0:  # make sure response is valid
                        gaze_class = responses[response_index][2]
                        gaze_labels.append(classes[gaze_class])
                        if not gaze_labels_only:
                            bbox = detect_face_opencv_dnn(net, frame, 0.7)
                            if not bbox:
                                no_face_counter += 1
                                face_labels.append(-2)
                                logging.info("[process_lkt_legacy] Video %s: Face not detected in frame %d" %
                                             (video_file.name, frame_counter))
                            else:
                                # select lowest face, probably belongs to kid: face = min(bbox, key=lambda x: x[3] - x[1])
                                selected_face = 0
                                min_value = bbox[0][3] - bbox[0][1]
                                # gaze_class = responses[response_index][2]
                                for i, face in enumerate(bbox):
                                    if bbox[i][3] - bbox[i][1] < min_value:
                                        min_value = bbox[i][3] - bbox[i][1]
                                        selected_face = i
                                    crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                                    # resized_img = cv2.resize(crop_img, (100, 100))
                                    resized_img = crop_img  # do not lose information in pre-processing step!
                                    face_box = np.array([face[1], face[1] + face[3], face[0], face[0] + face[2]])
                                    img_shape = np.array(frame.shape)
                                    ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                                                      face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
                                    face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
                                    face_ver = (ratio[0] + ratio[1]) / 2
                                    face_hor = (ratio[2] + ratio[3]) / 2
                                    face_height = ratio[1] - ratio[0]
                                    face_width = ratio[3] - ratio[2]
                                    feature_dict = {
                                        'face_box': face_box,
                                        'img_shape': img_shape,
                                        'face_size': face_size,
                                        'face_ver': face_ver,
                                        'face_hor': face_hor,
                                        'face_height': face_height,
                                        'face_width': face_width
                                    }
                                    img_filename = img_folder / f'{frame_counter:05d}_{i:01d}.png'
                                    if not img_filename.is_file() or force_create:
                                        cv2.imwrite(str(img_filename), resized_img)
                                    box_filename = box_folder / f'{frame_counter:05d}_{i:01d}.npy'
                                    if not box_filename.is_file() or force_create:
                                        np.save(str(box_filename), feature_dict)
                                valid_counter += 1
                                face_labels.append(selected_face)
                                # logging.info(f"valid frame in class {gaze_class}")
                    else:
                        no_annotation_counter += 1
                        gaze_labels.append(-2)
                        face_labels.append(-2)
                        logging.info("[process_lkt_legacy] Skipping since frame is invalid")
                else:
                    no_annotation_counter += 1
                    gaze_labels.append(-2)
                    face_labels.append(-2)
                    logging.info("[process_lkt_legacy] Skipping since no annotation (yet)")
            else:
                gaze_labels.append(-2)
                face_labels.append(-2)
                no_annotation_counter += 1
                logging.info("[process_lkt_legacy] Skipping frame since parser reported no annotation")
            ret_val, frame = cap.read()
            frame_counter += 1
            logging.info("[process_lkt_legacy] Processing frame: {}".format(frame_counter))
        # save gaze labels
        gaze_labels = np.array(gaze_labels)
        gaze_labels_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')
        if not gaze_labels_filename.is_file() or force_create:
            np.save(str(gaze_labels_filename), gaze_labels)
        if not gaze_labels_only:
            # save face labels
            face_labels = np.array(face_labels)
            face_labels_filename = Path.joinpath(args.faces_folder, video_file.stem, 'face_labels.npy')
            if not face_labels_filename.is_file() or force_create:
                np.save(str(face_labels_filename), face_labels)
        logging.info(
            "[process_lkt_legacy] Total frame: {}, No face: {}, No annotation: {}, Valid: {}".format(
                frame_counter, no_face_counter, no_annotation_counter, valid_counter))
        ed_time = time.time()
        logging.info('[process_lkt_legacy] Time used: %.2f sec' % (ed_time - st_time))


def generate_second_gaze_labels(force_create=False, visualize_confusion=True):
    """
    Processes the second annotator labels
    :param force_create: forces creation of files even if they exist
    :param visualize_confusion: if true, visualizes confusion between human annotators.
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    video_list = list(args.video_folder.glob("*.mp4"))
    for video_file in video_list:
        logging.info("[gen_2nd_labels] Video: %s" % video_file.name)
        if (args.label2_folder / (video_file.stem + '.txt')).exists():
            cap = cv2.VideoCapture(str(video_file))
            responses = parse_lookit_label(args.label2_folder / (video_file.stem + '.txt'), cap.get(cv2.CAP_PROP_FPS))
            gaze_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')))
            gaze_labels_second = []
            for frame in range(gaze_labels.shape[0]):
                if frame >= responses[0][0]:
                    q = [index for index, val in enumerate(responses) if frame >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1] != 0:
                        gaze_class = responses[response_index][2]
                        gaze_labels_second.append(classes[gaze_class])
                    else:
                        gaze_labels_second.append(-2)
                else:
                    gaze_labels_second.append(-2)
            gaze_labels_second = np.array(gaze_labels_second)
            gaze_labels_second_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels_second.npy')
            if not gaze_labels_second_filename.is_file() or force_create:
                np.save(str(gaze_labels_second_filename), gaze_labels_second)
        else:
            logging.info('[gen_2nd_labels] No label!')
    if visualize_confusion:
        visualize_human_confusion_matrix()


def visualize_human_confusion_matrix():
    """
    wrapper for calculating and visualizing confusion matrix with human annotations
    :return:
    """
    labels = []
    preds = []
    video_list = list(args.video_folder.glob("*.mp4"))
    for video_file in video_list:
        gaze_labels_second_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels_second.npy')
        if gaze_labels_second_filename.is_file():
            gaze_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')))
            gaze_labels_second = np.load(str(gaze_labels_second_filename))
            idxs = np.where((gaze_labels >= 0) & (gaze_labels_second >= 0))
            labels.extend(list(gaze_labels[idxs]))
            preds.extend(list(gaze_labels_second[idxs]))
    human_dir = Path('plots', 'human')
    human_dir.mkdir(exist_ok=True, parents=True)
    _, _ = visualize.calculate_confusion_matrix(labels, preds, human_dir / 'conf.pdf')


def gen_lookit_multi_face_subset(force_create=False):
    """
    Generates multi-face labels for training the face classifier
    :param force_create: forces creation of files even if they exist
    :return:
    """
    args.multi_face_folder.mkdir()
    names = [f.stem for f in Path(args.video_folder).glob('*.mp4')]
    face_hist = np.zeros(10)
    total_datapoint = 0
    for name in names:
        logging.info(name)
        src_folder = args.faces_folder / name
        dst_folder = args.multi_face_folder / name
        dst_folder.mkdir()
        (dst_folder / 'img').mkdir()
        (dst_folder / 'box').mkdir()
        face_labels = np.load(src_folder / 'face_labels.npy')
        files = list((src_folder / 'img').glob(f'*.png'))
        filenames = [f.stem for f in files]
        filenames = sorted(filenames)
        num_datapoint = 0
        for i in range(len(filenames)):
            if filenames[i][-1] != '0':
                face_label = face_labels[int(filenames[i][:5])]
                num_datapoint += 1
                total_datapoint += 1
                faces = [filenames[i - 1]]
                while i < len(filenames) and filenames[i][-1] != '0':
                    faces.append(filenames[i])
                    i += 1
                face_hist[len(faces)] += 1
                for face in faces:
                    dst_face_file = (dst_folder / 'img' / f'{name}_{face}_{face_label}.png')
                    if not dst_face_file.is_file() or force_create:
                        shutil.copy((src_folder / 'img' / (face + '.png')), dst_face_file)
                    dst_box_file = (dst_folder / 'box' / f'{name}_{face}_{face_label}.npy')
                    if not dst_box_file.is_file() or force_create:
                        shutil.copy((src_folder / 'box' / (face + '.npy')), dst_box_file)
        logging.info('# multi-face datapoint:{}'.format(num_datapoint))
    logging.info('total # multi-face datapoint:{}'.format(total_datapoint))
    logging.info(face_hist)


def process_dataset_face_classifier(args, force_create=False):
    """
    further process a dataset using a trained face baby vs adult classifier and nearest patch mechanism
    :param args: comman line arguments
    :param force_create: forces creation of files even if they exist
    :return:
    """

    ## todo: remove test code from here
    # val_infant_files = [f.stem for f in (face_data_folder / 'val' / 'infant').glob('*.png')]
    # val_others_files = [f.stem for f in (face_data_folder / 'val' / 'others').glob('*.png')]
    # num_correct = 0
    # total = len(val_infant_files) + len(val_others_files)
    # for f in val_infant_files:
    #     if f[-1] == f[-3]:
    #         num_correct += 1
    # for f in val_others_files:
    #     if f[-1] != f[-3]:
    #         num_correct += 1
    # logging.info("\n[process_lkt] {}, {}, {}".format(num_correct, total, num_correct / total))

    # emulate command line arguments
    # replace with what was used to train the face classifier!
    class Args:
        def __init__(self):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.rotation = False
            self.cropping = False
            self.hor_flip = False
            self.ver_flip = False
            self.color = False
            self.erasing = False
            self.noise = False
            self.model = "vgg16"
            self.dropout = 0.0

    fc_args = Args()
    model, input_size = face_classifier.fc_model.init_face_classifier(fc_args, model_name=fc_args.model, num_classes=2, resume_from=args.face_classifier_model_file)
    data_transforms = face_classifier.fc_eval.get_fc_data_transforms(fc_args, input_size)
    ## todo: remove test code from here
    # dataloaders = face_classifier.fc_data.get_dataset_dataloaders(args, input_size, 64, False)
    # criterion = face_classifier.fc_model.get_loss()
    # model.to(args.device)
    #
    # val_loss, val_top1, val_labels, val_probs, val_target_labels = face_classifier.fc_eval.evaluate(args,
    #                                                                                                 model,
    #                                                                                                 dataloaders['val'],
    #                                                                                                 criterion,
    #                                                                                                 return_prob=False,
    #                                                                                                 is_labelled=True,
    #                                                                                                 generate_labels=True)
    #
    # logging.info("\n[val] Failed images:\n")
    # err_idxs = np.where(np.array(val_labels) != np.array(val_target_labels))[0]
    # visualize.print_data_img_name(dataloaders, 'val', err_idxs)
    # logging.info('val_loss: {:.4f}, val_top1: {:.4f}'.format(val_loss, val_top1))
    model.to(args.device)
    model.eval()
    video_files = sorted(list(args.video_folder.glob("*.mp4")))
    for video_file in tqdm(video_files):
        face_labels_fc_filename = Path.joinpath(args.faces_folder, video_file.stem, 'face_labels_fc.npy')
        if not face_labels_fc_filename.is_file() or force_create:
            logging.info(video_file.stem)
            files = list((args.faces_folder / video_file.stem / 'img').glob(f'*.png'))
            filenames = [f.stem for f in files]
            filenames = sorted(filenames)
            idx = 0
            face_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'face_labels.npy')))
            face_labels_fc = []
            hor, ver = 0.5, 1
            for frame in tqdm(range(face_labels.shape[0])):
                if face_labels[frame] < 0:
                    face_labels_fc.append(face_labels[frame])
                else:
                    faces = []
                    centers = []
                    while idx < len(filenames) and (int(filenames[idx][:5]) == frame):
                        img = Image.open(args.faces_folder / video_file.stem / 'img' / (filenames[idx] + '.png')).convert(
                            'RGB')
                        box = np.load(args.faces_folder / video_file.stem / 'box' / (filenames[idx] + '.npy'),
                                      allow_pickle=True).item()
                        centers.append([box['face_hor'], box['face_ver']])
                        img = data_transforms['val'](img)
                        faces.append(img)
                        idx += 1
                    centers = np.stack(centers)
                    faces = torch.stack(faces).to(args.device)
                    output = model(faces)
                    _, preds = torch.max(output, 1)
                    preds = preds.cpu().numpy()
                    idxs = np.where(preds == 0)[0]
                    centers = centers[idxs]
                    if centers.shape[0] == 0:
                        face_labels_fc.append(-1)
                    else:
                        dis = np.sqrt((centers[:, 0] - hor) ** 2 + (centers[:, 1] - ver) ** 2)
                        i = np.argmin(dis)
                        face_labels_fc.append(idxs[i])
                        hor, ver = centers[i]
            face_labels_fc = np.array(face_labels_fc)
            np.save(str(face_labels_fc_filename), face_labels_fc)


def report_dataset_split():
    """
    prints out a list of training and test videos according to the heuristic that doubly coded videos are test set.
    :return:
    """
    all_videos = []
    test_videos = []
    for path in sorted(args.label_folder.glob("*.txt")):
        all_videos.append(path.name)
    for path in sorted(args.label2_folder.glob("*.txt")):
        test_videos.append(path.name)
    train_videos = [x for x in all_videos if x not in test_videos]
    logging.info("train videos: [{0}]".format(', '.join(map(str, train_videos))))
    logging.info("test videos: [{0}]".format(', '.join(map(str, test_videos))))
    # visualize_human_confusion_matrix()


if __name__ == "__main__":
    args = options.parse_arguments_for_preprocess()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())

    if args.raw_dataset_type == "lookit":
        preprocess_raw_lookit_dataset(args, force_create=False)
    elif args.raw_dataset_type == "princeton":
        preprocess_raw_princeton_dataset(args, force_create=False)
    elif args.raw_dataset_type == "generic":
        preprocess_raw_generic_dataset(args, force_create=False)
    else:
        raise NotImplementedError

    process_dataset_lowest_face(args, gaze_labels_only=False, force_create=False)
    generate_second_gaze_labels(force_create=False, visualize_confusion=False)
    # report_dataset_split()
    # gen_lookit_multi_face_subset(force_create=False)
    # uncomment next line if face classifier was trained:
    process_dataset_face_classifier(args, force_create=False)
