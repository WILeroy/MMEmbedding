import os
import json
import pandas as pd
import tqdm


videoid2url = {}
videoid2cap = {}

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    print(df.columns.tolist())

    videoids = df['video_id'].values.tolist()
    urls = df['video_url'].values.tolist()
    captions = df['video_caption'].values.tolist()

    print(len(videoids))
    print(len(urls))
    
    for video_id, url, caption in zip(videoids, urls, captions):
        video_id = str(video_id)
        videoid2url[video_id] = url
        videoid2cap[video_id] = caption

""" 需要修改两个全量索引文件路径.
"""
load_csv('../../data/shopee-video/meta/shopee_video_20220224_video_caption_quanliang.csv')
load_csv('../../data/shopee-video/meta/shopee_video_20220216_video_caption.csv')

print(len(list(videoid2url.keys())))

""" 需要修改两个全量索引文件路径.
"""
log_dir = '../../data/shopee-video/prelogs/'
frame_dir = '../../data/shopee-video/frames/'
audio_dir = '../../data/shopee-video/audios/'


if __name__ == '__main__':
    dataset = {}
    
    for vid in tqdm.tqdm(videoid2url.keys()):
        log_at = os.path.join(log_dir, vid)
        frame_at = os.path.join(frame_dir, vid)
        audio_at = os.path.join(audio_dir, vid+'.wav')

        if os.path.exists(log_at):
            with open(log_at) as f:
                log_content = f.read()

        if log_content in ['download_wrong', 'frame_wrong_audio_wrong', 'frame_wrong_audio_done']:
            print(vid, 'error')
            continue

        if log_content == 'frame_done_audio_done':                    
            if os.path.exists(frame_at) and os.path.exists(audio_at):
                if type(videoid2cap[vid]) is float:
                    videoid2cap[vid] = ''   
                dataset[vid] = {'frames':frame_at,
                                'audio':audio_at,
                                'caption':videoid2cap[vid],
                                'url':videoid2url[vid]}
            else:
                print('log error')
        elif log_content == 'frame_done_audio_wrong':
            if os.path.exists(frame_at):
                if type(videoid2cap[vid]) is float:
                    videoid2cap[vid] = ''
                dataset[vid] = {'frames':frame_at,
                                'audio':'empty',
                                'caption':videoid2cap[vid],
                                'url':videoid2url[vid]}
            else:
                print('log error')
        else:
            print('log dont existed', log_content, vid)
    
    print(len(list(dataset.keys())))

    with open('allset.json', 'w') as f:
        json.dump(dataset, f, indent=4, separators=[',', ':'])
