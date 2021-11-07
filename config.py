from pathlib import Path

classes = {"away": 0, "left": 1, "right": 2}
face_model_file = Path("models", "face_model.caffemodel")
model_path = Path("/tmp/pycharm_project_628/models/yotam.pth")
# model_path = Path("/tmp/pycharm_project_628/models/latest_net.pth")

config_file = Path("models", "config.prototxt")
raw_dataset_folder = Path("/home/jupyter/data/lookIt_raw")
processed_data_set_folder = Path("/home/jupyter/data/lookIt")
video_folder = raw_dataset_folder / "videos"
faces_folder = processed_data_set_folder / "faces"
label_folder = processed_data_set_folder / "coding_first"
label2_folder = processed_data_set_folder / "coding_second"
multi_face_folder = processed_data_set_folder / "multi_face"
face_data_folder = processed_data_set_folder / "infant_vs_others"
visualization_root = Path("/home/jupyter/deployment")
inference_output = Path("/home/jupyter/inference_output")
inference_output_pre_fine_tune = Path("/home/jupyter/inference_output_pre_fine_tune")
inference_output_post_fine_tune = Path("/home/jupyter/inference_output_post_fine_tune")

mini_marchman = True
if mini_marchman:
    marchman_root = Path("/home/jupyter/osfstorage-archive/TL3 Study/18m")
    video_folder_A = marchman_root / "Visit A/Mov"
    video_folder_B = marchman_root / "Visit B/Mov"
    label_folder_A = marchman_root / "Visit A/VCX"
    label_folder_B = marchman_root / "Visit B/VCX"
