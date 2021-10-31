import cv2
from pathlib import Path
import numpy as np
from models import GazeCodingModel, MyModule
from preprocess import detect_face_opencv_dnn
from options import parse_arguments_for_testing
import visualize
import logging


def prep_frame(popped_frame, bbox, class_text, face):
    popped_frame = visualize.put_text(popped_frame, class_text)
    if bbox:
        popped_frame = visualize.put_rectangle(popped_frame, face)
        if not class_text == "away" and not class_text == "off" and not class_text == "on":
            popped_frame = visualize.put_arrow(popped_frame, class_text, face)
    return popped_frame


def predict_from_video(opt):
    # initialize
    import torch
    logging.info("using the following values for per-channel mean: {}".format(opt.per_channel_mean))
    logging.info("using the following values for per-channel std: {}".format(opt.per_channel_std))
    face_model_file = Path("models", "face_model.caffemodel")
    config_file = Path("models", "config.prototxt")
    path_to_primary_model = opt.model
    opt.frames_per_datapoint = 10
    opt.frames_stride_size = 2
    primary_model = GazeCodingModel(opt).to(opt.device)
    if opt.device == 'cpu':
        primary_model.load_state_dict(torch.load(str(path_to_primary_model), map_location=torch.device(opt.device)))
    else:
        primary_model.load_state_dict(torch.load(str(path_to_primary_model)))
    primary_model.eval()
    resize_window = 100
    classes = {'away': 0, 'left': 1, 'right': 2}
    reverse_dict = {0: 'away', 1: 'left', 2: 'right'}
    sequence_length = 9
    loc = -5
    # load face extractor model
    face_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    # set video source
    if opt.source_type == 'file':
        video_path = Path(opt.source)
        if video_path.is_dir():
            logging.warning("Video folder provided as source. Make sure it contains video files only.")
            video_paths = list(video_path.glob("*"))
            video_paths = [str(x) for x in video_paths]
        elif video_path.is_file():
            video_paths = [str(video_path)]
        else:
            raise NotImplementedError
    else:
        video_paths = [int(opt.source)]
    for i in range(len(video_paths)):
        video_path = Path(str(video_paths[i]))
        answers = []
        image_sequence = []
        box_sequence = []
        frames = []
        frame_count = 0
        last_class_text = ""  # Initialize so that we see the first class assignment as an event to record
        logging.info("predicting on : {}".format(video_paths[i]))
        cap = cv2.VideoCapture(video_paths[i])
        # Get some basic info about the video
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolution = (int(width), int(height))
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        # If creating annotated video output, set up now
        if opt.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # may need to be adjusted per available codecs & OS
            my_suffix = video_path.suffix
            if not my_suffix:
                my_suffix = ".mp4"
            my_video_path = Path(opt.output_video_path, video_path.stem + "_output{}".format(my_suffix))
            video_output = cv2.VideoWriter(str(my_video_path), fourcc, framerate, resolution, True)
        if opt.output_annotation:
            my_output_file_path = Path(opt.output_annotation, video_path.stem + "_annotation.txt")
            output_file = open(my_output_file_path, "w", newline="")
            if opt.output_format == "PrefLookTimestamp":
                # Write header
                output_file.write(
                    "Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")
        # iterate over frames
        ret_val, frame = cap.read()
        img_shape = np.array(frame.shape)
        while ret_val:
            frames.append(frame)
            bbox = detect_face_opencv_dnn(face_model, frame, 0.7)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
            if not bbox:
                answers.append(classes['away'])  # if face detector fails, treat as away and mark invalid
                image = np.zeros((1, resize_window, resize_window, 3), np.float64)
                my_box = np.array([0, 0, 0, 0, 0])
                box_sequence.append(my_box)
                image_sequence.append((image, True))
                face = None
            else:
                # todo: improve face selection mechanism
                face = min(bbox, key=lambda x: x[3] - x[1])  # select lowest face in image, probably belongs to kid
                crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]

                if crop_img.size == 0:
                    answers.append(classes['away'])  # if face detector fails, treat as away and mark invalid
                    image = np.zeros((1, resize_window, resize_window, 3), np.float64)
                    image_sequence.append((image, True))
                    my_box = np.array([0, 0, 0, 0, 0])
                    box_sequence.append(my_box)
                else:
                    answers.append(classes['left'])  # if face detector succeeds, treat as left and mark valid
                    image = cv2.resize(crop_img, (resize_window, resize_window)) * 1. / 255
                    image = np.expand_dims(image, axis=0)
                    image -= np.array(opt.per_channel_mean)
                    image /= (np.array(opt.per_channel_std) + 1e-6)
                    image_sequence.append((image, False))
                    face_box = np.array([face[1], face[1] + face[3], face[0], face[0] + face[2]])
                    ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                                      face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
                    face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
                    face_ver = (ratio[0] + ratio[1]) / 2
                    face_hor = (ratio[2] + ratio[3]) / 2
                    face_height = ratio[1] - ratio[0]
                    face_width = ratio[3] - ratio[2]
                    my_box = np.array([face_size, face_ver, face_hor, face_height, face_width])
                    box_sequence.append(my_box)
            if len(image_sequence) == sequence_length:
                popped_frame = frames[loc]
                frames.pop(0)
                if not image_sequence[sequence_length // 2][1]:  # if middle image is valid
                    if opt.model == "icatcher":
                        to_predict = [x[0] for x in image_sequence[0::2]]
                        prediction = primary_model.predict(to_predict)
                        predicted_classes = np.argmax(prediction, axis=1)
                        int32_pred = np.int32(predicted_classes[0]).item()
                    elif opt.model == "icatcher+":
                        to_predict = {"imgs": torch.tensor([x[0] for x in image_sequence[0::2]], dtype=torch.float).squeeze().permute(0, 3, 1, 2),
                                      "boxs": torch.tensor(box_sequence[::2], dtype=torch.float)
                                      }
                        with torch.set_grad_enabled(False):
                            outputs = primary_model(to_predict)
                            _, prediction = torch.max(outputs, 1)
                            int32_pred = prediction.cpu().numpy()[0]
                    answers[loc] = int32_pred
                image_sequence.pop(0)
                box_sequence.pop(0)
                class_text = reverse_dict[answers[-sequence_length]]
                if opt.on_off:
                    class_text = "off" if class_text == "away" else "on"
                # If show_output or output_video is true, add text label, bounding box for face, and arrow showing direction
                if opt.show_output:
                    prepped_frame = prep_frame(popped_frame, bbox, class_text, face)
                    cv2.imshow('frame', prepped_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if opt.output_video_path:
                    prepped_frame = prep_frame(popped_frame, bbox, class_text, face)
                    video_output.write(prepped_frame)

                if opt.output_annotation:
                    if opt.output_format == "raw_output":
                        output_file.write("{}, {}\n".format(frame_count, class_text))
                    elif opt.output_format == "PrefLookTimestamp":
                        if class_text != last_class_text: # Record "event" for change of direction if code has changed
                            frame_ms = int(1000. / framerate * frame_count)
                            output_file.write("{},0,{}\n".format(frame_ms, class_text))
                            last_class_text = class_text
                    else:
                        raise NotImplementedError
                logging.info("frame: {}, class: {}".format(str(frame_count - sequence_length + 1), class_text))
            ret_val, frame = cap.read()
            frame_count += 1
        if opt.show_output:
            cv2.destroyAllWindows()
        if opt.output_video_path:
            video_output.release()
        if opt.output_annotation:  # write footer to file
            if opt.output_format == "PrefLookTimestamp":
                frame_ms = int(1000. / framerate * frame_count)
                output_file.write("0,{},codingactive\n".format(frame_ms))
                output_file.close()
        cap.release()


if __name__ == '__main__':
    args = parse_arguments_for_testing()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    predict_from_video(args)
