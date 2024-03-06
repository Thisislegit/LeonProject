import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np
import torch.nn.functional as F

# create dataset
class LeonDataset(Dataset):
    def __init__(self, labels, costs1, costs2, nodes1=[], nodes2=[], nodeFeaturizer=None, dict=None):
        self.labels = labels
        self.costs1 = costs1
        self.costs2 = costs2
        self.nodeFeaturizer = nodeFeaturizer
        self.dict = dict
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        assert self.nodes1[0].info.get('all_filters_est_rows') is not None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
    
        target_cols = 200
        trees1 = self.dict[self.nodes1[idx].info['index']][0].squeeze(0)
        padding_cols = max(0, target_cols - trees1.size(1))
        trees1 = F.pad(trees1, (0, padding_cols), value=0)
        indexes1 = self.dict[self.nodes1[idx].info['index']][1].squeeze(0)
        padding_cols = max(0, target_cols - indexes1.size(0))
        indexes1 = F.pad(indexes1, (0, 0, 0, padding_cols), value=0)
        trees2 = self.dict[self.nodes2[idx].info['index']][0].squeeze(0)
        padding_cols = max(0, target_cols - trees2.size(1))
        trees2 = F.pad(trees2, (0, padding_cols), value=0)
        indexes2 = self.dict[self.nodes2[idx].info['index']][1].squeeze(0)
        padding_cols = max(0, target_cols - indexes2.size(0))
        indexes2 = F.pad(indexes2, (0, 0, 0, padding_cols), value=0)
        query_feats1 = self.nodes1[idx].info['query_feature']
        query_feats2 = self.nodes2[idx].info['query_feature']
        vecs1 = self.dict[self.nodes1[idx].info['index']][2].squeeze(0)
        vecs2 = self.dict[self.nodes2[idx].info['index']][2].squeeze(0)
        return {
            'labels': self.labels[idx],
            'costs1': self.costs1[idx],
            'costs2': self.costs2[idx],
            'encoded_plans1': trees1,
            'encoded_plans2': trees2,
            'attns1': indexes1,
            'attns2': indexes2,
            'queryfeature1': query_feats1,
            'queryfeature2': query_feats2,
            'vecs1': vecs1,
            'vecs2': vecs2
        }
    
def prepare_dataset(pairs, Shouldquery, nodeFeaturizer, dict=None):
    labels = []
    costs1 = []
    costs2 = []
    Nodes1 = []
    Nodes2 = []
    for pair in pairs:
        if pair[0].info['latency'] > pair[1].info['latency']:
            label = 0
        else:
            label = 1
        if Shouldquery:
            Nodes1.append(
                pair[0]
            )
            Nodes2.append(
                pair[1]
            )
        labels.append(label)
        costs1.append(pair[0].cost)
        costs2.append(pair[1].cost)
    labels = torch.tensor(labels)
    costs1 = torch.tensor(costs1)
    costs2 = torch.tensor(costs2)
    dataset = LeonDataset(labels, costs1, costs2, Nodes1, Nodes2, nodeFeaturizer, dict)
    return dataset


class BucketDataset(Dataset):
    def __init__(self, buckets, keys=None, nodeFeaturizer=None, dict=None):
        # filter buckets with in keys
        if keys:
            buckets = {key: value for key, value in buckets.items() if value and key in keys}
        else:
            buckets = {key: value for key, value in buckets.items() if value}
        self.buckets_dict = buckets
        
        self.buckets = list(buckets.values())
        # Flatten buckets
        self.buckets_item = [item for bucket in self.buckets_dict.values() for item in bucket]
        self.nodeFeaturizer = nodeFeaturizer
        self.dict = dict
        

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)

    def __getitem__(self, idx):
        
        node, b, c = self.buckets_item[idx]
        # null_node = plans_lib.Binarize(node)
        target_cols = 200
        trees = self.dict[node.info['index']][0].squeeze(0)
        padding_cols = max(0, target_cols - trees.size(1))
        trees = F.pad(trees, (0, padding_cols), value=0)
        indexes = self.dict[node.info['index']][1].squeeze(0)
        padding_cols = max(0, target_cols - indexes.size(0))
        indexes = F.pad(indexes, (0, 0, 0, padding_cols), value=0)
        # trees, indexes = encoding.TreeConvFeaturize(self.nodeFeaturizer, [null_node])
        item = {'join_tables': node.info['join_tables'], \
                'plan_encode': trees, \
                'att_encode': indexes, \
                'latency': node.info['latency'], \
                'cost': node.cost,\
                'sql': node.info['sql_str'], \
                'queryfeature': node.info['query_feature']}
        return item

class BucketBatchSampler(Sampler):
    def __init__(self, buckets, batch_size):
        self.buckets = buckets
        self.bucket_indices = list(range(len(buckets)))
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.bucket_indices)
        for i in range(0, len(self.bucket_indices), self.batch_size):
            yield [item for bucket_idx in self.bucket_indices[i:i+self.batch_size] for item in range(sum(len(bucket) for bucket in self.buckets[:bucket_idx]), sum(len(bucket) for bucket in self.buckets[:bucket_idx+1]))]
            
    def __len__(self):
        return len(self.bucket_indices) // self.batch_size
    
    
class PreTrainDataset(torch.utils.data.Dataset):
    """A dataset of execution plans and associated costs."""

    def __init__(self,
                 query_feats,
                 plans,
                 indexes,
                 costs,
                 tree_conv=False,
                 transform_cost=True,
                 label_mean=None,
                 label_std=None,
                 cross_entropy=False,
                 return_indexes=True):
        """Dataset of plans/parent positions/costs.

        Args:
          query_feats: a list of np.ndarray (float).
          plans: a list of np.ndarray (int64).
          indexes: a list of np.ndarray (int64).
          costs: a list of floats.
          transform_cost (optional): if True, log and standardize.
        """
        assert len(plans) == len(costs) and len(query_feats) == len(plans)

        query_feats = [torch.from_numpy(xs) for xs in query_feats]
        if not tree_conv:
            plans = [torch.from_numpy(xs) for xs in plans]
            if return_indexes:
                assert len(plans) == len(indexes)
                indexes = [torch.from_numpy(xs) for xs in indexes]

        self.query_feats = query_feats
        self.plans = plans
        self.indexes = indexes

        if not isinstance(transform_cost, list):
            transform_cost = [transform_cost]
        self.transform_cost = transform_cost
        self.cross_entropy = cross_entropy
        self.return_indexes = return_indexes

        self.label_mean = label_mean
        self.label_std = label_std

        if cross_entropy:
            # Classification.

            # We don't care too much about a diff of a few millis, so scale the
            # raw millis values by this factor.
            #
            # 0.1: roughly, distinguish every 10ms.
            # 0.01: roughly, distinguish every 100ms.
            # 0.001: roughly, distinguish every second.
            self.MS_SCALE_FACTOR = 1  # millis * factor.
            self.MS_SCALE_FACTOR = 1e-2  # millis * factor.
            self.MS_SCALE_FACTOR = 1e-1  # millis * factor.

            costs = np.asarray(costs) * self.MS_SCALE_FACTOR
            # Use invertible transform on scalar x:
            #     x -> sqrt(x + 1) - 1 + transform_eps * x
            # where transform_eps is a param.
            self.TRANSFORM_EPS = 1e-3

            costs = np.sqrt(costs + 1.) - 1 + self.TRANSFORM_EPS * costs
            self.costs = torch.as_tensor(costs).to(torch.float32)
            print('transformed costs, min', costs.min(), 'max', costs.max())
        else:
            # Regression.
            for t in transform_cost:
                fn = self._transform_fn(t)
                costs = fn(costs)
            self.costs = torch.as_tensor(costs).to(torch.float)

    def _transform_fn(self, transform_name):

        def log1p(xs):
            return np.log(np.asarray(xs) + 1.0)

        def standardize(xs):
            self._EPS = 1e-6
            if self.label_mean is None:
                self.mean = np.mean(xs)
                self.std = np.std(xs)
            else:
                self.mean = self.label_mean
                self.std = self.label_std
            print('costs stats mean {} std {}'.format(self.mean, self.std))
            return (xs - self.mean) / (self.std + self._EPS)

        def min_max(xs):
            self.label_min = np.min(xs)
            self.label_max = np.max(xs)
            self.label_range = self.label_max - self.label_min
            print('costs stats min {} max {}'.format(self.label_min,
                                                     self.label_max))
            return (xs - self.label_min) / self.label_range

        transforms = {
            'log1p': log1p,
            True: standardize,
            'standardize': standardize,
            False: (lambda xs: xs),
            'min_max': min_max,
            'sqrt': lambda xs: (np.sqrt(1 + np.asarray(xs))),
        }
        return transforms[transform_name]

    def _inverse_transform_fn(self, transform_name, use_torch=False):

        def log1p_inverse(xs):
            return np.exp(xs) - 1.0

        def log1p_inverse_torch(xs):
            return torch.exp(xs) - 1.0

        def standardize_inverse(xs):
            return xs * (self.std + self._EPS) + self.mean

        def min_max_inverse(xs):
            return xs * self.label_range + self.label_min

        transforms = {
            'log1p': log1p_inverse,
            True: standardize_inverse,
            'standardize': standardize_inverse,
            False: (lambda xs: xs),
            'min_max': min_max_inverse,
            'sqrt': lambda xs: (xs**2 - 1),
        }
        if use_torch:
            transforms['log1p'] = log1p_inverse_torch
        return transforms[transform_name]

    def InvertCost(self, cost):
        """Convert model outputs back to latency space."""
        if self.cross_entropy:
            with torch.no_grad():
                softmax = torch.softmax(torch.from_numpy(cost), -1)
                expected = (torch.arange(softmax.shape[-1]) * softmax).sum(-1)

                # Inverse transform.
                # predicted_latency = ((expected + 1)**2 - 1).numpy()
                # return predicted_latency

                e = expected.numpy()
                assert self.TRANSFORM_EPS == 1e-3
                x = 1e3 * (e - np.sqrt(1e3 * e + 251001) + 501)
                return x / self.MS_SCALE_FACTOR
        else:
            for t in reversed(self.transform_cost):
                fn = self._inverse_transform_fn(t)
                cost = fn(cost)
            return cost

    def TorchInvertCost(self, cost):
        """Convert model outputs back to latency space."""
        assert not self.cross_entropy, 'Not implemented'
        for t in reversed(self.transform_cost):
            fn = self._inverse_transform_fn(t, use_torch=True)
            cost = fn(cost)
        return cost

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        if self.return_indexes:
            return self.query_feats[idx], self.plans[idx], self.indexes[
                idx], self.costs[idx]
        return self.query_feats[idx], self.plans[idx], self.costs[idx]

    def FreeData(self):
        self.query_feats = self.plans = self.indexes = self.costs = None
