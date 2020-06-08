class Fusion:    

    def _concat_feat(self, feat1, feat2):
        print(feat1)
        print(feat2)
        samples = feat1
        delete_idx = []
        for idx in range(len(samples)):
            for feat in feat2:
                feat = self._to_dict(feat)
                key = samples[idx]['img']
                if key not in feat:
                    delete_idx.append(idx)
                    continue
                assert feat[key]['cls'] == samples[idx]['cls']
                samples[idx]['hist'] = np.append(samples[idx]['hist'], feat[key]['hist'])
        for d_idx in sorted(set(delete_idx), reverse=True):
            del samples[d_idx]
        if delete_idx != []:
            print("Ignore %d samples" % len(set(delete_idx)))

        return samples

    def _to_dict(self, feat):
        print(feat)
        ret = {}
        for f in feat:
            ret[f['img']] = {
                'cls': f['cls'],
                'hist': f['hist']
            }
        return ret
