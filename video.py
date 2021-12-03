import ffmpeg
import logging


def get_video_meta_data(video_file_path):
    probe = ffmpeg.probe(str(video_file_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info


def verify_constant_framerate(video_file_path):
    meta_data = get_video_meta_data(video_file_path)
    r_fps = int(meta_data['r_frame_rate'].split('/')[0]) / int(meta_data['r_frame_rate'].split('/')[1])
    avg_fps = int(meta_data['avg_frame_rate'].split('/')[0]) / int(meta_data['avg_frame_rate'].split('/')[1])
    if r_fps != avg_fps:
        logging.warning("video fps is not constant")
        print(meta_data)


def get_fps(video_file_path):
    meta_data = get_video_meta_data(video_file_path)
    return int(meta_data['r_frame_rate'].split('/')[0]) / int(meta_data['r_frame_rate'].split('/')[1])