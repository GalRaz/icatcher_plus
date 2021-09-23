from pathlib import Path

classes = {"away": 0, "left": 1, "right": 2}
face_model_file = Path("models", "face_model.caffemodel")
config_file = Path("models", "config.prototxt")
raw_folder = Path("datasets", "lookit_raw")
processed_data_set_folder = Path("datasets", "lookit")
video_folder = processed_data_set_folder / "videos"
label_folder = processed_data_set_folder / "coding_first"
label2_folder = processed_data_set_folder / "coding_second"
dataset_folder = processed_data_set_folder / "dataset"
multi_face_folder = processed_data_set_folder / "multi_face"
face_data_folder = processed_data_set_folder / "infant_vs_others"
