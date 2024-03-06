import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import plans_lib

DEVICE = 'cpu'


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = TreeConv1d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = TreeConv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # self.shortcut = nn.Sequential(
            #     nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3),
            #     nn.BatchNorm1d(out_channels)
            # )
            self.shortcut = TreeConv1d(in_channels, out_channels)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        trees, indexes = x
        residual = trees

        out, indexes = self.conv1((trees, indexes))
        out = self.bn1(out)
        out = self.relu(out)
        out, indexes = self.conv2((out, indexes))
        out = self.bn2(out)
        residual, index = self.shortcut((residual, indexes))
        residual = self.shortcut_bn(residual)

        out += residual
        out = F.relu(out)
        return out, indexes

class TreeConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.

    Value is either cost or latency.
    """

    def __init__(self, feature_size, plan_size, label_size, version=None):
        super(TreeConvolution, self).__init__()
        # None: default
        assert version is None, version
        self.query_p = 0.3
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        self.conv = nn.Sequential(
            TreeConv1d(32 + plan_size, 512),
            # TreeConv1d(plan_size, 512),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 256),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(256, 128),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),
        )
        self.plan_p = 0.2
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )
        # self.reset_weights()
        self.apply(self._init_weights)
        self.model_type = "TreeConv"

    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query_feats, trees, indexes):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors.  Shaped as
            [batch size, query dims].
          trees: The input plan features.  Shaped as
            [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        # Give larger dropout to query features.
        query_embs = nn.functional.dropout(query_feats, p=0.5)
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        #    print(query_embs.shape)
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)

        out = self.conv((concat, indexes))
        out = self.out_mlp(out)
        return out

class TreeResnetConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.

    Value is either cost or latency.
    """

    def __init__(self, feature_size, plan_size, label_size, version=None):
        super(TreeResnetConvolution, self).__init__()
        # None: default
        assert version is None, version
        self.query_p = 0.3
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )

        self.conv = nn.Sequential(TreeConv1d(32 + plan_size, 64, kernel_size=3, padding=1, stride=1),
                    TreeStandardize(),
                    TreeAct(nn.LeakyReLU()),
                    TreeMaxPool_With_Kernel_Stride(),
                    *resnet_block(64, 64, 2, first_block=True),
                    *resnet_block(64, 128, 2),
                    TreeAvgPool())
        self.plan_p = 0.2
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )
        # self.reset_weights()
        self.apply(self._init_weights)
        self.model_type = "TreeConv"

    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query_feats, trees, indexes):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors.  Shaped as
            [batch size, query dims].
          trees: The input plan features.  Shaped as
            [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        # Give larger dropout to query features.
        query_embs = nn.functional.dropout(query_feats, p=0.5)
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        #    print(query_embs.shape)
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)

        out = self.conv((concat, indexes))
        out = self.out_mlp(out)
        return out

class TreeConv1d(nn.Module):
    """Conv1d adapted to tree data."""

    def __init__(self, in_dims, out_dims, kernel_size=3, stride=3, padding=0):
        super().__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = self.weights(
            torch.gather(data, 2,
                         indexes.expand(-1, -1, self._in_dims).transpose(1, 2)))
        zeros = torch.zeros((data.shape[0], self._out_dims)).unsqueeze(2).to(feats.device)
        feats = torch.cat((zeros, feats), dim=2)
        return feats, indexes


class TreeMaxPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].max(dim=2).values
    
class TreeAvgPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].mean(dim=2)

class TreeMaxPool_With_Kernel_Stride(nn.Module):

    def __init__(self, kernel_size=3, stride=3, padding=0):
        super(TreeMaxPool_With_Kernel_Stride, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return (F.max_pool1d(trees[0], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), trees[1])

class  TreeAct(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return self.activation(trees[0]), trees[1]


class TreeStandardize(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        mu = torch.mean(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        s = torch.std(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        standardized = (trees[0] - mu) / (s + 1e-5)
        return standardized, trees[1]

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = TreeConv1d(input_channels, num_channels,
                                         kernel_size=3, padding=1, stride=strides)
        self.conv2 = TreeConv1d(num_channels, num_channels,
                                         kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = TreeConv1d(input_channels, num_channels,
                                         kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = TreeStandardize()
        self.bn2 = TreeStandardize()
        self.relu1 = TreeAct(nn.LeakyReLU())
        self.relu2 = TreeAct(nn.LeakyReLU())

    def forward(self, X):
        Y = self.relu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu2(Y)
    
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet(nn.Module):
    def __init__(self, feature_size, plan_size, label_size, \
                 block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = plan_size + 32

        self.conv1 = TreeConv1d(self.in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.in_channels = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3])
        # self.linear = nn.Linear(512, num_classes)
        # self.max_pool = TreeMaxPool_With_Kernel_Stride()

        self.query_p = 0.3
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )

        # self.plan_p = 0.2
        self.out_mlp = nn.Sequential(
            nn.Linear(512, 32),
            # nn.Dropout(p=self.plan_p),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )

        self.tree_pool = TreeMaxPool()

        self.apply(self._init_weights)
        self.model_type = "TreeConv"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_layer(self, block, out_channels, num_blocks, stride=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, query_feats, trees, indexes):
        query_embs = nn.functional.dropout(query_feats, p=0.5)
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        #    print(query_embs.shape)
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)


        out, indexes = self.conv1((concat, indexes))
        out = self.bn1(out)
        out = F.relu(out)
        # out, indexes = self.max_pool((out, indexes))
        out, indexes = self.layer1((out, indexes))
        out, indexes = self.layer2((out, indexes))
        out, indexes = self.layer3((out, indexes))
        out, indexes = self.layer4((out, indexes))
        out = self.tree_pool((out, indexes))
        # out = out.view(out.size(0), -1)
        out = self.out_mlp(out)
        return out


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


# @profile
def _batch(data, padding_size):
    # lens = [vec.shape[0] for vec in data]
    # if len(set(lens)) == 1:
    #     # Common path.
    #     return np.asarray(data)
    xs = np.zeros((len(data), padding_size, data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs


# @profile
def _make_preorder_ids_tree(curr, root_index=1):
    """Returns a tuple containing a tree of preorder positional IDs.

    Returns (tree structure, largest id under me).  The tree structure itself
    (the first slot) is a 3-tuple:

    If curr is a leaf:
      tree structure is (my id, 0, 0) (note that valid IDs start with 1)
    Else:
      tree structure is
        (my id, tree structure for LHS, tree structure for RHS).

    This function traverses each node exactly once (i.e., O(n) time complexity).
    """
    if not curr.children:
        return (root_index, 0, 0), root_index
    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0],
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1],
                                              root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


# @profile
def _walk(curr, vecs):
    if curr[1] == 0:
        # curr is a leaf.
        vecs.append(curr)
    else:
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)
        _walk(curr[2], vecs)


# @profile
def _make_indexes(root):
    # Join(A, B) --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C) --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    preorder_ids, _ = _make_preorder_ids_tree(root)
    vecs = []
    _walk(preorder_ids, vecs)
    # Continuing with the Join(A,B) example:
    # Preorder traversal _walk() produces
    #   [1, 2, 3]
    #   [2, 0, 0]
    #   [3, 0, 0]
    # which would be reshaped into
    #   array([[1],
    #          [2],
    #          [3],
    #          [2],
    #          [0],
    #          [0],
    #    ...,
    #          [0]])
    vecs = np.asarray(vecs).reshape(-1, 1)
    return vecs


# @profile
def _featurize_tree(curr_node, node_featurizer):
    def _bottom_up(curr):
        """Calls node_featurizer on each node exactly once, bottom-up."""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            vec = node_featurizer.FeaturizeLeaf(curr)
            curr.__node_feature_vec = vec
            return vec
        left_vec = _bottom_up(curr.children[0])
        right_vec = _bottom_up(curr.children[1])
        vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)
    vecs = []
    plans_lib.MapNode(curr_node,
                      lambda node: vecs.append(node.__node_feature_vec))
    # Add a zero-vector at index 0.
    ret = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    ret[1:] = vecs
    return ret


# @profile
def make_and_featurize_trees(trees, node_featurizer, padding_size):
    indexes = torch.from_numpy(_batch([_make_indexes(x) for x in trees], padding_size)).long()
    trees = torch.from_numpy(
        _batch([_featurize_tree(x, node_featurizer) for x in trees
                ], padding_size)).transpose(1, 2)
    return trees, indexes

# @profile
def Premake_and_featurize_trees(trees, node_featurizer, padding_size):
    indexes = torch.from_numpy(np.asarray([_make_indexes(x) for x in trees])).long()
    trees = torch.from_numpy(
        np.asarray([_featurize_tree(x, node_featurizer) for x in trees
                ])).transpose(1, 2)
    return trees, indexes