import json
import os

import pandas as pd
import tqdm


videoid2url = {}
videoid2cap = {}
url2videoid = {}

def load_csv(csv_path):
    if csv_path.endswith('.tsv'):
        df = pd.read_csv(csv_path, sep='\t')        
    else:
        df = pd.read_csv(csv_path)

    print(csv_path, 'columns:', df.columns.tolist())

    videoids = df['video_id'].values.tolist()
    try:
        urls = df['video_url'].values.tolist()
    except:
        urls = df['play_url'].values.tolist()
    captions = df['video_caption'].values.tolist()

    print(csv_path, 'video num:', len(videoids))
    
    for video_id, url, caption in zip(videoids, urls, captions):
        video_id = str(video_id)
        videoid2url[video_id] = url
        url2videoid[url] = video_id
        videoid2cap[video_id] = caption

def load_allset():
    """ 需要修改全量索引文件路径. """
    load_csv('../../data/shopee-video/meta/seedset_90.tsv')
    load_csv('../../data/shopee-video/meta/shopee_video_20220224_video_caption_quanliang.csv')
    load_csv('../../data/shopee-video/meta/shopee_video_20220216_video_caption.csv')

    print('allset video num:', len(list(videoid2url.keys())))
    return videoid2url, videoid2cap, url2videoid


""" 需要修改两个全量索引文件路径.
"""
log_dir = '../../data/shopee-video/prelogs/'
frame_dir = '../../data/shopee-video/frames/'
audio_dir = '../../data/shopee-video/audios/'


if __name__ == '__main__':
    dataset = {}
    
    _, videoid2cap, _ = load_allset()

    for vid in tqdm.tqdm(videoid2cap.keys()):
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
    
    print('useful video num:', len(list(dataset.keys())))

    with open('data/allset.json', 'w') as f:
        json.dump(dataset, f, indent=4, separators=[',', ':'])
