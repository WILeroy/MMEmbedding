import argparse
import json
import os.path as osp
from multiprocessing import Pool

import numpy as np
import tqdm

import utils.visulize as visulize
from utils.gallery import Gallery


def run_top10(query_feat):
    global gallery
    
    y_score = gallery.cosines(query_feat)
    top10_indexes = np.argsort(-y_score)[1:10]
    top10_ids = [gallery.id(idx) for idx in top10_indexes]
    top10_scores = [y_score[idx] for idx in top10_indexes]

    return {'top10':top10_ids, 'top10_scores':top10_scores}


#################### global gallery ####################
gallery = Gallery()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_dir')
    parser.add_argument('target_data')
    parser.add_argument('all_data')
    parser.add_argument('--save-gallery', type=str, default=None)
    parser.add_argument('--reload', type=str, default=None)
    parser.add_argument('--save-result', type=str, default='shopee-video_results.html')
    args = parser.parse_args()

    with open(args.all_data) as f:
        allset = json.load(f)

    with open(args.target_data) as f:
        targetset = json.load(f)

    latestlabel = "-1"
    qids = []
    for k in targetset.keys():
        if targetset[k]['label'] != latestlabel:
            qids.append(k)
        latestlabel = targetset[k]['label']
        
    gids = list(allset.keys())

    print(len(qids))
    print(len(gids))

    if args.reload is not None:
        gallery.reload(args.reload, True)
    else:
        gallery.load_features(args.emb_dir, gids, True)

    if args.save_gallery is not None:
        gallery.save(args.save_gallery)

    pool = Pool(96)

    reqs = []
    for idx, qid in enumerate(qids):
        qemb = gallery._load_feat(osp.join(args.emb_dir, qid+'.npy'))
        reqs.append((idx, qid, pool.apply_async(run_top10, args=(qemb, ))))

    draw_data = []
    cnt = 0
    recall = 0
    for idx, qid, req in tqdm.tqdm(reqs):
        result = req.get()

        cosines = result['top10_scores']
        recall_ids = result['top10']
        recall_urls = []
        recall_captions = []
        recall_rights = []
        cntlocal = 0
        for rid in recall_ids:
            recall_urls.append(allset[rid]['url'])
            recall_captions.append(allset[rid]['caption'])

            if rid in targetset.keys():
                if targetset[rid]['label'] == targetset[qid]['label']:
                    print(targetset[rid]['label'], targetset[qid]['label'])
                    recall_rights.append(True)
                    cnt += 1
                    cntlocal += 1
                else:
                    recall_rights.append(False)                    
            else:
                recall_rights.append(False)

        recall += cntlocal * 1.0 / 9

        block_data = visulize.draw_video_to_videos(
            qid, allset[qid]['url'], allset[qid]['caption'], 
            recall_ids, cosines, recall_urls, recall_captions, recall_rights)

        draw_data.append(block_data)
        #if idx == 9: break

    visulize.draw_page(args.save_result, draw_data, 5)

    print(cnt)
    print(recall / 40)
