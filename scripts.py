import pandas as pd

def create_personalized_dataset(tsv_path):
    video_to_id_dict = {}
    df = pd.read_csv(tsv_path, sep="\t")
    id_list = df['childID']
    for id in id_list:
        if id not in video_to_id_dict.keys():
            video_to_id_dict[id] = []
        temp_df = df[df.childID == id]
        to_add_list = temp_df['videoID']
        for vid in to_add_list:
            video_to_id_dict[id].append(vid)

    return video_to_id_dict



print(create_personalized_dataset("prephys_split0_videos.tsv"))