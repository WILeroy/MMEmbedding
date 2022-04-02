import json
import os

import tqdm

from build_allset import load_allset


_, videoid2cap, _ = load_allset()

""" 需要修改视频帧和音频存储路径. """
log_dir = '../../data/shopee-video/prelogs/'
frame_dir = '../../data/shopee-video/frames/'
audio_dir = '../../data/shopee-video/audios/'


if __name__ == '__main__':
    dataset = {}
    label = 0
    
    """ 需要修改弱监督原始文件路径. """
    with open('../../data/shopee-video/meta/video_list_with_weak_pos.txt') as f:
        for line in tqdm.tqdm(f.readlines()):
            tokens = line.strip().split()
            query, pos1 = tokens[0].split(',')
            allvids = [query] + [pos1] + tokens[1:]
            
            for vid in allvids:
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
                            if vid in dataset.keys():
                                dataset[vid]['label'].append(label)
                            else:
                                if type(videoid2cap[vid]) is float:
                                    videoid2cap[vid] = ''   
                                dataset[vid] = {'frames':frame_at,
                                                'audio':audio_at,
                                                'caption':videoid2cap[vid],
                                                'label':[label]}
                        else:
                            print('log error')
                    elif log_content == 'frame_done_audio_wrong':
                        if os.path.exists(frame_at):
                            if vid in dataset.keys():
                                dataset[vid]['label'].append(label)
                            else:
                                if type(videoid2cap[vid]) is float:
                                    videoid2cap[vid] = ''   
                                dataset[vid] = {'frames':frame_at,
                                                'audio':'empty',
                                                'caption':videoid2cap[vid],
                                                'label':[label]}
                        else:
                            print('log error')
                else:
                    print('log dont existed', log_content, vid)

            label += 1
    
    print('useful video num:', len(list(dataset.keys())))

    with open('data/weak_trainset.json', 'w') as f:
        json.dump(dataset, f, indent=4, separators=[',', ':'])
