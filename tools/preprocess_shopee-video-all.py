import os
import time
from multiprocessing import Pool

import requests
import tqdm

from build_allset import load_allset


videoid2url, _, _ = load_allset()

""" 存储数据的路径. """
log_dir = '../../data/shopee-video/prelogs/'
frame_dir = '../../data/shopee-video/frames/'
audio_dir = '../../data/shopee-video/audios/'
tmp_dir = '../../data/shopee-video/tmp'


def download_video(idx, vid, url, log_save_to):
    if os.path.exists(log_save_to):
        with open(log_save_to) as f:
            log_concent = f.read()
            if log_concent == 'download_wrong':
                print('bad url', vid, videoid2url[vid])
            elif log_concent == 'frame_wrong_audio_wrong':
                print('bad video', vid, videoid2url[vid])
            elif log_concent == 'frame_wrong_audio_done':
                print('only bad frame', vid, videoid2url[vid])
            elif log_concent in ['frame_done_audio_wrong', 'frame_done_audio_done']:
                """ resume. """
                return
            else:
                print('can\'t check', log_concent)
        print(vid, 'try again')                
    else:
        print('no log', vid, videoid2url[vid])
        print(vid, 'try again')

    results = ''

    frames_save_to_dir = os.path.join(frame_dir, vid)
    if not os.path.exists(frames_save_to_dir): os.mkdir(frames_save_to_dir)
        
    tmp_save_to = os.path.join(tmp_dir, vid+'.mp4')
    frames_save_to = os.path.join(frames_save_to_dir, 'frame_%05d.jpg')
    audio_save_to = os.path.join(audio_dir, f'{vid}.wav')
        
    t = time.time()
    response = requests.get(url, stream=True)
    content = response.content
    t1 = time.time() - t

    if response.status_code != 200:
        results = f'download_wrong'
    else:
        with open(tmp_save_to, 'wb') as f:
            f.write(content)
        #-nostats -loglevel 0
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

        os.system(f'rm {tmp_save_to}')
                    
        results += 'frame_done' if frames_success == 0 else 'frame_wrong'
        results += '_audio_done' if audio_success == 0 else '_audio_wrong'

        if idx % 1000 == 0: print(time.ctime(), idx, t1, t2, t3)

    with open(log_save_to, 'w') as f:
        f.write(results)
        

if __name__ == '__main__':
    pool = Pool(64)

    cnt = 1
    for videoid in tqdm.tqdm(videoid2url.keys()):
        log_save_to = os.path.join(log_dir, videoid)
        pool.apply_async(download_video, args=(cnt, videoid, videoid2url[videoid], log_save_to))
        cnt += 1

    pool.close()
    pool.join()
    