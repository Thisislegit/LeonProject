import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from torch.nn.functional import pad
import json
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F

configs = {
    'node_length' : 18, # +1 for ???Path  # of node_types +2 
    'pad_length' : 50,
    # training
    "loss_weight" : 0.5,
    'max_runtime' : 1,
    'query_dim' : 666 # 666
}

plan_parameters = [
    "Node Type",
    "Total Cost",
    "Plan Rows",
]

def load_json(path):
    with open(path) as json_file:
        json_obj = json.load(json_file)
    return json_obj


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v["type"] == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v["center"]
            scaler.scale_ = v["scale"]
            feature_statistics[k]["scaler"] = scaler


class FeatureType(Enum):
    numeric = "numeric"
    categorical = "categorical"

    def __str__(self):
        return self.value


def dfs(plan, seq, adjs, parent_node_id, run_times, heights, cur_height, if_train):
    cur_node_id = len(seq)
    seq.append(plan)
    heights.append(cur_height)
    if if_train:
        run_times.append(plan["Actual Total Time"])
    else:
        run_times.append(-1)
    if parent_node_id != -1:  # not root node
        adjs.append((parent_node_id, cur_node_id))
    if "Plans" in plan:
        for child in plan["Plans"]:
            dfs(child, seq, adjs, cur_node_id, run_times, heights, cur_height + 1, if_train)


def scale_feature(feature_statistics, feature, node):
    if feature_statistics[feature]["type"] == str(FeatureType.numeric):
        scaler = feature_statistics[feature]["scaler"]
        return scaler.transform(np.array([node[feature]]).reshape(-1, 1))
    else:
        return feature_statistics[feature]["value_dict"][node["Node Type"]]


def pad_sequence(seq_encoding, padding_value=0, node_length=18, max_length=40):
    """
    pad seqs to the same length, and transform seqs to a tensor
    """
    # seqs: list of seqs (seq's shape: (1, feature_no)))
    # padding_value: padding value
    # return: padded seqs, seqs_length
    seq_length = seq_encoding.shape[1]
    seq_padded = pad(
        torch.from_numpy(seq_encoding),
        (0, max_length * node_length - seq_encoding.shape[1]),
        value=padding_value,
    )
    seq_padded = seq_padded.to(dtype=torch.float32)
    return seq_padded, seq_length


def generate_seqs_encoding(
    seq, op_name_to_one_hot, plan_parameters, feature_statistics
):
    seq_encoding = []
    for node in seq:
        # add op_name encoding
        op_name = node[plan_parameters[0]]
        op_encoding = op_name_to_one_hot[op_name]
        seq_encoding.append(op_encoding)
        # add other features, and scale them
        for feature in plan_parameters[1:]:
            feature_encoding = scale_feature(feature_statistics, feature, node)
            seq_encoding.append(feature_encoding)
    seq_encoding = np.concatenate(seq_encoding, axis=1)

    return seq_encoding


def get_plan_sequence(plan, pad_length=40, if_train=False):
    """
    plan: a plan read from json file
    pad_length: int, the length of the padded seqs (the number of nodes in the plan)

    return: seq, run_times, adjs, heights, database_id
    seq: List, each element is a node's plan_parameters
    run_times: List, each element is a node's runtime
    adjs: List, each element is a tuple of (parent, child)
    heights: List, each element is a node's height
    database_id: int, the id of the database
    """
    # get all sub-plans' runtime
    seq = []
    run_times = []
    adjs = []  # [(parent, child)]
    heights = []  # the height of each node, root node's height is 0
    dfs(plan["Plan"], seq, adjs, -1, run_times, heights, 0, if_train)
    # padding run_times to the same length
    if len(run_times) < pad_length:
        run_times = run_times + [1] * (pad_length - len(run_times))
    return seq, run_times, adjs, heights, 0 # 0 is database_id


# get attention mask
def get_attention_mask(adj, seq_length, pad_length, node_length):
    # adjs: List, each element is a tuple of (parent, child)
    # seqs_length: List, each element is the length of a seq
    # pad_length: int, the length of the padded seqs
    # return: attention mask
    seq_length = int(seq_length / node_length)

    attention_mask_seq = np.ones((pad_length, pad_length))
    for a in adj:
        attention_mask_seq[a[0], a[1]] = 0

    # based on the reachability of the graph, set the attention mask
    for i in range(seq_length):
        for j in range(seq_length):
            if attention_mask_seq[i, j] == 0:
                for k in range(seq_length):
                    if attention_mask_seq[j, k] == 0:
                        attention_mask_seq[i, k] = 0

    # node can reach itself
    for i in range(pad_length):
        attention_mask_seq[i, i] = 0

    # to tensor
    attention_mask_seq = torch.tensor(attention_mask_seq, dtype=torch.bool)
    return attention_mask_seq


def get_loss_mask(seq_length, pad_length, node_length, height, loss_weight=0.5):
    seq_length = int(seq_length / node_length)
    loss_mask = np.zeros((pad_length))
    loss_mask[:seq_length] = np.power(loss_weight, np.array(height))
    loss_mask = torch.from_numpy(loss_mask).float()
    return loss_mask


# get a plan's encoding
def get_plan_encoding(
    plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics):
    """
    plan: a plan read from json file
    pad_length: int, the length of the padded seqs (the number of nodes in the plan)
    """
    seq, run_times, adjs, heights, database_id = get_plan_sequence(
        plan, configs["pad_length"])
    run_times = np.array(run_times).astype(np.float32) / configs["max_runtime"] + 1e-7
    run_times = torch.from_numpy(run_times)
    seq_encoding = generate_seqs_encoding(
        seq, op_name_to_one_hot, plan_parameters, feature_statistics
    )

    # pad seq_encoding
    seq_encoding, seq_length = pad_sequence(
        seq_encoding,
        padding_value=0,
        node_length=configs["node_length"],
        max_length=configs["pad_length"],
    )

    # get attention mask
    attention_mask = get_attention_mask(
        adjs, seq_length, configs["pad_length"], configs["node_length"]
    )

    # get loss mask
    # only for training
    # loss_mask = get_loss_mask(
    #     seq_length,
    #     configs["pad_length"],
    #     configs["node_length"],
    #     heights,
    #     configs["loss_weight"],
    # )

    return seq_encoding, run_times, attention_mask, None, database_id # database_id无用


# create SeqFormer
class SeqFormer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.1,
        transformer_dropout=0.1,
        query_dim=None,
        node_embedding_dim=None,
        padding_size=40
    ):
        super(SeqFormer, self).__init__()
        # input_dim: node bits
        self.node_length_before = input_dim

        embedding_size = 0
        if node_embedding_dim is not None:
            embedding_size += node_embedding_dim + 2
        if embedding_size == 0:
            embedding_size += input_dim
        if query_dim is not None:
            embedding_size += 32
        
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_size,
                dim_feedforward=hidden_dim,
                nhead=1,
                batch_first=True,
                activation=transformer_activation,
                dropout=transformer_dropout,
            ),
            num_layers=1,
        )
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        self.mlp_hidden_dims = [128, 64, 32]
        # self.mlp_hidden_dims = [256, 128, 1]
        self.mlp = nn.Sequential(
            *[
                nn.Linear(embedding_size, self.mlp_hidden_dims[0]),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                nn.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1]),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                nn.Linear(self.mlp_hidden_dims[1], output_dim),
            ]
        )

        if query_dim is not None:
            self.query_mlp = nn.Sequential(
                nn.Linear(query_dim, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
            )
        self.node_embedding_dim = node_embedding_dim
        if node_embedding_dim:
            self.node_embedding = nn.Embedding(16, node_embedding_dim)
            self.Norm = nn.BatchNorm1d(padding_size)
        self.apply(self._init_weights)
        self.model_type = "Transformer"
        # self.sigmoid = nn.Sigmoid()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attn_mask=None, queryfeature=None):
        # queryfeature = None
        # query_feats: Query encoding vectors.  Shaped as [batch size, query dims].
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 18 bits
        x = x.view(x.shape[0], -1, self.node_length_before)

        if self.node_embedding_dim:
            node_part = x[:, :, :16]
            node_part = torch.argmax(node_part, dim=2)
            stats_part = x[:, :, 16:]
            x_node = self.node_embedding(node_part.long())
            x = torch.cat((x_node, stats_part), axis=2)

        if queryfeature is not None:
            query_embs = self.query_mlp(queryfeature.unsqueeze(1))
            max_subtrees = x.shape[1]
            query_embs = query_embs.expand(query_embs.shape[0], max_subtrees, 
                                           query_embs.shape[2])
            x = torch.cat((query_embs, x), axis=2)


        # attn_mask = attn_mask.repeat(4,1,1)
        if self.node_embedding_dim and queryfeature is not None:
            x = self.Norm(x)
        out = self.tranformer_encoder(x, mask=attn_mask)
        # out = self.transformer_decoder(out, out, tgt_mask=attn_mask)
        out = self.mlp(out)
        # out = (torch.tanh(out).squeeze(dim=2) * 5).add(5) 
        # out = torch.tanh(out).squeeze(dim=2).add(1) * 5
        out = torch.tanh(out).squeeze(dim=2).add(1)
        return out # [0, 1] -> [1, 2] [??]

# 树形编码结构
class SeqFormer_tree(nn.Module):
    def __init__(
        self,
        query_dim,
        input_dim,
        hidden_dim,
        output_dim,
        padding_dim,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.3,
        transformer_dropout=0.2,
    ):
        super(SeqFormer_tree, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                dim_feedforward=hidden_dim,
                nhead=1,
                batch_first=True,
                activation=transformer_activation,
                dropout=transformer_dropout,
            ),
            num_layers=1,
        )
        self.node_length = input_dim
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        # self.mlp_hidden_dims = [128, 64, 32]
        self.mlp_hidden_dims = [128, 64, 1]
        self.mlp = nn.Sequential(
            *[
                nn.Linear(self.node_length, self.mlp_hidden_dims[0]),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                nn.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1]),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                nn.Linear(self.mlp_hidden_dims[1], output_dim),
            ]
        )
        self.sigmoid = nn.Sigmoid()

        self.query_mlp = nn.Sequential(
            nn.Linear(query_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        self.padding_dim = padding_dim

    def forward(self, query_feats, trees, indexes, attn_mask=None):

        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)
        gather = torch.gather(concat.transpose(1, 2), 1, indexes.expand(-1, -1, 155))
        # (batch, seq_len, input_size)
        # padding
        padding_length = self.padding_dim - gather.size(1)
        x = F.pad(gather, (0, 0, 0, padding_length, 0, 0))

        # attn_mask = attn_mask.repeat(4,1,1)
        out = self.tranformer_encoder(x, mask=attn_mask)
        # out = self.transformer_decoder(out, out, tgt_mask=attn_mask)
        out = self.mlp(out)
        out = self.sigmoid(out).squeeze(dim=2)
        return out * 2 # [0, 1] -> [1, 2] [??]

def get_op_name_to_one_hot(feature_statistics):
    op_name_to_one_hot = {}
    op_names = feature_statistics["node_types"]["value_dict"]
    op_names_no = len(op_names)
    for i, name in enumerate(op_names.keys()):
        op_name_to_one_hot[name] = np.zeros((1, op_names_no), dtype=np.int32)
        op_name_to_one_hot[name][0][i] = 1
    return op_name_to_one_hot


if __name__ == "__main__":

    statistics_file_path = "data/zero_shot_filted/statistics.json"

    feature_statistics = load_json(statistics_file_path)
    # add numerical scalers (cite from zero-shot)
    add_numerical_scalers(feature_statistics)

    # create SeqFormer model
    model = SeqFormer(
        input_dim=36,
        hidden_dim=128,
        output_dim=1,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.3,
        transformer_dropout=0.2,
    )
