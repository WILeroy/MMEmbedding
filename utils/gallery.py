import os.path as osp

import h5py
import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class Gallery():
    def __init__(self):
        self.features = None
        self.ids = []

    def id(self, index):
        return self.ids[index]

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

    def load_features(self, feature_dir, ids, verbose=False):
        self.features = []
        for gid in tqdm.tqdm(ids):
            feat = self._load_feat(osp.join(feature_dir, gid+'.npy'), verbose)
            if feat is not None:
                self.features.append(feat)
                self.ids.append(gid)

        if verbose: print(f'Gallery.load_features: loaded {len(self.features)}')
        
        self.features = np.stack(self.features)

    def save(self, save_to):
        h5f = h5py.File(save_to, 'w')
        h5f.create_dataset('features', data=self.features)
        h5f.create_dataset('ids', data=[id.encode('utf-8') for id in self.ids])

    def reload(self, save_at, verbose=False):
        h5f = h5py.File(save_at)
        self.features = h5f['features']
        self.ids = [id.decode('utf-8') for id in h5f['ids']]

        if verbose: print(f'Gallery.reload: loaded {self.features.shape[0]}')
