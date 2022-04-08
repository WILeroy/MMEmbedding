import json
import time
import os
import multiprocessing as mp

import pandas as pd


""" 存储数据的路径. """
log_dir = '../../data/tiktok/prelogs/'
frame_dir = '../../data/tiktok/frames/'
audio_dir = '../../data/tiktok/audios/'
tmp_dir = '../../data/tiktok/videos'


def preprocess_video(idx, vid, log_save_to):
    """if os.path.exists(log_save_to):
        with open(log_save_to) as f:
            log_concent = f.read()
            if log_concent == 'download_wrong':
                print('bad url', vid)
            elif log_concent == 'frame_wrong_audio_wrong':
                print('bad video', vid)
            elif log_concent == 'frame_wrong_audio_done':
                print('only bad frame', vid)
            elif log_concent in ['frame_done_audio_wrong', 'frame_done_audio_done']:
                #resume
                return
            else:
                print('can\'t check', log_concent)
        print(vid, 'try again')                
    else:
        print('no log', vid)
        print(vid, 'try again')"""

    results = ''

    frames_save_to_dir = os.path.join(frame_dir, vid)
    if not os.path.exists(frames_save_to_dir): os.mkdir(frames_save_to_dir)
        
    tmp_save_to = os.path.join(tmp_dir, vid+'.mp4')

    if os.path.exists(tmp_save_to):
        frames_save_to = os.path.join(frames_save_to_dir, 'frame_%05d.jpg')
        audio_save_to = os.path.join(audio_dir, f'{vid}.wav')
        
        cmd_extract_frames = 'ffmpeg -nostats -loglevel 0 -i {} -y -vf "scale=256*iw/min(iw\,ih):256*ih/min(iw\,ih),fps=fps=2" -q:v 2 {}'.format(
            tmp_save_to, frames_save_to
        )
        cmd_extract_audio = 'ffmpeg -nostats -loglevel 0 -i {} -y -f wav -ar 16000 -ac 1 {}'.format(
            tmp_save_to, audio_save_to
        )

        t = time.time()
        frames_success = os.system(cmd_extract_frames)
        t2 = time.time() - t

        t = time.time()
        audio_success = os.system(cmd_extract_audio)
        t3 = time.time() - t
                    
        results += 'frame_done' if frames_success == 0 else 'frame_wrong'
        results += '_audio_done' if audio_success == 0 else '_audio_wrong'
    else:
        results = 'download_wrong'

    if idx % 1000 == 0: print(time.ctime(), idx, t2, t3)

    with open(log_save_to, 'w') as f:
        f.write(results)


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
    url_template = '../videos/{}.mp4'
    cnt = 0

    pool = mp.Pool(64)

    for kword in list(keyword2rank.keys()):
        qtext = kword
        gids, gurls, gtexts, greds, gauthors = [], [], [], [], []
        for video in keyword2rank[kword]:
            #print(video)
            log_save_to = os.path.join(log_dir, video['video_id'])
            pool.apply_async(preprocess_video, args=(cnt, video['video_id'], log_save_to))
            cnt += 1
        
    pool.close()
    pool.join()