import json
from types import SimpleNamespace
import numpy as np
from enum import Enum
from sklearn.preprocessing import RobustScaler


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(
                json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def dfs_adj(plan, seq, adjs, parent_node_id):
    cur_node_id = len(seq)
    seq.append(plan)
    if parent_node_id != -1:  # not root node
        adjs.append((parent_node_id, cur_node_id))
    if 'Plans' in plan:
        for child in plan['Plans']:
            dfs_adj(child, seq, adjs, cur_node_id)
    elif 'SubPlans' in plan:
        for child in plan['SubPlans']:
            dfs_adj(child, seq, adjs, cur_node_id)


def get_plan_seq_adj(plan):
    seq = []
    adjs = []  # [(parent, child)]
    dfs_adj(plan, seq, adjs, -1)
    return seq, adjs


class FeatureType(Enum):
    numeric = "numeric"
    categorical = "categorical"

    def __str__(self):
        return self.value


def scale_feature(feature_statistics, feature, node):
    if feature_statistics[feature]["type"] == str(FeatureType.numeric):
        scaler = feature_statistics[feature]["scaler"]
        return scaler.transform(np.array([getattr(node, feature)]).reshape(-1, 1))
    else:
        return feature_statistics[feature]["value_dict"][node.op_name]


def generate_seqs_encoding(seqs):
    seqs_encoding = []
    for seq in seqs:
        seq_encoding = []
        for node in seq:
            # add op_name encoding
            op_name = node['Node Type']
            if op_name == '???Path':
                op_name = 'SeqScan'
            op_encoding = op_name_to_one_hot[op_name]
            seq_encoding.append(op_encoding)
            # add other features, and scale them
            for feature in plan_parameters[1:]:
                feature_encoding = scale_feature(
                    feature_statistics, feature, node)
                seq_encoding.append(feature_encoding)
        seq_encoding = np.concatenate(seq_encoding, axis=1)
        seqs_encoding.append(seq_encoding)

    return seqs_encoding


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v.get("type") == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v["center"]
            scaler.scale_ = v["scale"]
            feature_statistics[k]["scaler"] = scaler


# get seqs encoding
plan_parameters = [
    "Node Type",
    # "est_startup_cost",
    "Total Cost",
    "Plan Rows",
    # "est_width",
    # "est_children_card",
    # "workers_planned",
]
# statistics_file_path = "/data1/liangzibo/zero-shot/zero-shot-data/runs/parsed_plans/statistics_workload_combined.json"
# feature_statistics = load_json(statistics_file_path, namespace=False)
# op_name_to_one_hot = {}
# op_names = feature_statistics["op_name"]["value_dict"]
# op_names_no = len(op_names)
# for i, name in enumerate(op_names.keys()):
#     op_name_to_one_hot[name] = np.zeros((1, op_names_no), dtype=np.int32)
#     op_name_to_one_hot[name][0][i] = 1

# add_numerical_scalers(feature_statistics)

# node type
'''
{'Nested Loop Anti Join': 0, 'Parallel Seq Scan': 1, 'Index Only Scan': 2, 'Parallel Bitmap Heap Scan': 3, 'Seq Scan': 4, 'Gather Merge': 5, 'Partial Aggregate': 6, 'Aggregate': 7, 'Sort': 8, 'Parallel Hash Join': 9, 'Parallel Hash Anti Join': 10, 'Merge Left Join': 11, 'Nested Loop Left Join': 12, 'Bitmap Heap Scan': 13, 'Hash': 14, 'Materialize': 15, 'Parallel Index Only Scan': 16, 'Hash Left Join': 17, 'Nested Loop': 18, 'Finalize Aggregate': 19, 'Parallel Hash Left Join': 20, 'Bitmap Index Scan': 21, 'Gather': 22, 'Parallel Hash': 23, 'Hash Right Join': 24, 'Hash Join': 25, 'Index Scan': 26, 'Parallel Index Scan': 27, 'Merge Join': 28, 'Merge Right Join': 29}
'''
