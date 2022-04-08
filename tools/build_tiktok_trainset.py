import json
import time
import os
import multiprocessing as mp
import tqdm

import pandas as pd


""" 存储数据的路径. """
log_dir = '../../data/tiktok/prelogs/'
frame_dir = '../../data/tiktok/frames/'
audio_dir = '../../data/tiktok/audios/'
tmp_dir = '../../data/tiktok/videos'


def load_ranks(csv_path):
    df = pd.read_csv(csv_path)
    print(df.columns)

    contents = df["content"].to_list()
    keywords = df["keyword"].to_list()
    print(len(contents), len(keywords))
    keywords2rank = {}
    for idx, kword in enumerate(keywords):
        content = contents[idx]
        items = json.loads(content)
        keywords2rank[kword] = items

    return keywords2rank

if __name__ == '__main__':
    keyword2rank = load_ranks('../../data/tiktok/meta/tt_query_20220309_raw.csv')
    draw_data = []
    label = 0
    dataset = {}

    for kword in tqdm.tqdm(list(keyword2rank.keys())):
        for video in keyword2rank[kword]:
            if video['video_id'] in dataset.keys():
                #print(video['video_id'], 'existed')
                dataset[video['video_id']]['label'].append(label)
                dataset[video['video_id']]['rank'].append(video['rank'])
            else:
                frame_at = os.path.join(frame_dir, video['video_id'])
                audio_at = os.path.join(audio_dir, video['video_id']+'.wav')
                log_at = os.path.join(log_dir, video['video_id'])

                if os.path.exists(log_at):
                    with open(log_at) as f:
                        log_content = f.read()
                    
                    if log_content in ['download_wrong', 'frame_wrong_audio_wrong', 'frame_wrong_audio_done']:
                        #print(video['video_id'], log_content)
                        continue

                    if log_content == 'frame_done_audio_done':                    
                        if os.path.exists(frame_at) and os.path.exists(audio_at):
                            if type(video['caption']) is float:
                                video['caption'] = ''   
                            dataset[video['video_id']] = {
                                'frames':frame_at,
                                'audio':audio_at,
                                'caption':video['caption'],
                                'label':[label],
                                'rank':[video['rank']]}
                        else:
                            print('log error')
                    elif log_content == 'frame_done_audio_wrong':
                        if os.path.exists(frame_at) and os.path.exists(audio_at):
                            if type(video['caption']) is float:
                                video['caption'] = ''   
                            dataset[video['video_id']] = {
                                'frames':frame_at,
                                'audio':'empty',
                                'caption':video['caption'],
                                'label':[label],
                                'rank':[video['rank']]}
                        else:
                            print('log error')
                else:
                    print(log_at, 'dont existed')
        label += 1

    print(len(list(dataset.keys())))

    with open('data/tiktok_trainset.json', 'w') as f:
        json.dump(dataset, f, indent=4, separators=[',', ':'])