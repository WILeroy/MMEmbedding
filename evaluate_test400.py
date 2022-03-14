import argparse
import os.path as osp
from multiprocessing import Pool
import json
import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import utils.visulize as visulize


class Gallery():
    def __init__(self, feature_dim):
        self.feat_dim = feature_dim
        self.features = None
        self.index_of_id = {}
        self.ids = []

    def index(self, id):
        return self.index_of_id[id]

    def id(self, index):
        return self.ids[index]

    def url(self, index):
        return self.urls[index]

    def caption(self, index):
        return self.captions[index]

    def cosines(self, query_feature):
        if len(query_feature.shape) != 2:
            query_feature = query_feature[np.newaxis, :]

        if self.features is None:
            print('Gallery.cosines: features is None.')
            return None
        return cosine_similarity(query_feature, self.features)[0]
    
    def _load_feat(self, fpath, verbose=False):
        try:
            feat = np.load(fpath, allow_pickle=True)
        except:
            if verbose: 
                print(f'Gallery._load_feat: {fpath} empty')
            feat = None
        
        return feat

    def load_features(self, features, ids, urls, captions, verbose):
        self.features = features
        self.ids = ids
        self.urls = urls
        self.captions = captions
        self.index_of_id = {}
        for index, id in enumerate(self.ids): 
            self.index_of_id[id] = index

        if verbose: print(f'Gallery.load_fetures: loaded {len(self.ids)}, {self.features.shape}')

    def load_features_floder(self, feature_dir, ids, verbose=False):
        self.features = []
        cnt = 0
        for gid in tqdm.tqdm(ids):
            feat = self._load_feat(osp.join(feature_dir, gid+'.npy'), verbose)
            if feat is not None:
                self.features.append(feat)
                self.index_of_id[gid] = cnt
                cnt += 1

        if verbose: print(f'Gallery.load_features: loaded {len(self.features)}')
        
        self.ids = list(self.index_of_id.keys())
        self.features = np.stack(self.features)


def run_top10(query_feat):
    global gallery
    
    y_score = gallery.cosines(query_feat)
    top10_indexes = np.argsort(-y_score)[1:10]
    top10_ids = [gallery.id(idx) for idx in top10_indexes]
    #top10_urls = [gallery.url(idx) for idx in top10_indexes]
    #top10_captions = [gallery.caption(idx) for idx in top10_indexes]
    top10_scores = [y_score[idx] for idx in top10_indexes]

    return {'top10':top10_ids, 'top10_scores':top10_scores}#, 'top10_urls':top10_urls, 'top10_captions':top10_captions}


#################### global gallery ####################
gallery = Gallery(512)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_dir')
    parser.add_argument('target_data')
    parser.add_argument('--save-to', type=str, default='shopee-video_results.html')
    args = parser.parse_args()

    with open(args.target_data) as f:
        dataset = json.load(f)

    latestlabel = "-1"
    qids = []
    gids = []
    for k in dataset.keys():
        if dataset[k]['label'] != latestlabel:
            qids.append(k)
        else:
            gids.append(k)
        latestlabel = dataset[k]['label']
        
    print(len(qids))
    print(len(gids))

    gallery.load_features_floder(args.emb_dir, gids, True)

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
            recall_urls.append(dataset[rid]['url'])
            recall_captions.append(dataset[rid]['caption'])
            if dataset[rid]['label'] == dataset[qid]['label']:
                recall_rights.append(True)
                cnt += 1
                cntlocal += 1
            else:
                recall_rights.append(False)

        recall += cntlocal * 1.0 / 9

        block_data = visulize.draw_video_to_videos(
            qid, dataset[qid]['url'], dataset[qid]['caption'], 
            recall_ids, cosines, recall_urls, recall_captions, recall_rights)

        draw_data.append(block_data)
        #if idx == 9: break

    visulize.draw_page(args.save_to, draw_data, 5)

    print(cnt)
    print(recall / 40)