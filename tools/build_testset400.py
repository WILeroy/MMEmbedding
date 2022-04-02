import json
import os

from build_allset import load_allset


_, videoid2cap, _ = load_allset()

""" 需要修改视频帧和音频存储路径. """
log_dir = '../../data/shopee-video/prelogs/'
frame_dir = '../../data/shopee-video/frames/'
audio_dir = '../../data/shopee-video/audios/'


if __name__ == '__main__':
    dataset = {}

    """ 需要修改400测试原始文件路径. """
    with open('../../data/shopee-video/meta/testset_400.tsv') as f:
        for line in f.readlines():
            tokens = line.strip().split('\t')
            cate = tokens[0]
            videoid = tokens[2]

            if videoid in videoid2cap.keys():
                log_at = os.path.join(log_dir, videoid)
                frame_at = os.path.join(frame_dir, videoid)
                audio_at = os.path.join(audio_dir, videoid+'.wav')

                if os.path.exists(log_at):
                    with open(log_at) as f:
                        log_content = f.read()

                    if log_content in ['download_wrong', 'frame_wrong_audio_wrong', 'frame_wrong_audio_done']:
                        print(videoid, 'error')
                        continue

                    if log_content == 'frame_done_audio_done':                    
                        if os.path.exists(frame_at) and os.path.exists(audio_at):
                            if type(videoid2cap[videoid]) is float:
                                videoid2cap[videoid] = ''   
                            dataset[videoid] = {'frames':frame_at,
                                                'audio':audio_at,
                                                'caption':videoid2cap[videoid],
                                                'label':cate}
                        else:
                            print('log error')
                    elif log_content == 'frame_done_audio_wrong':
                        if os.path.exists(frame_at):
                            if type(videoid2cap[videoid]) is float:
                                videoid2cap[videoid] = ''   
                            dataset[videoid] = {'frames':frame_at,
                                                'audio':'empty',
                                                'caption':videoid2cap[videoid],
                                                'label':cate}
                        else:
                            print('log error')
                else:
                    print('log dont existed', log_content, videoid)
    
    print('useful video num:', len(list(dataset.keys())))

    with open('data/testset400.json', 'w') as f:
        json.dump(dataset, f, indent=4, separators=[',', ':'])
