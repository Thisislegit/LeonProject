import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import copy

# Encoding channels for each plan node
channels = ['EstNodeCost', 'EstRows', 'EstBytes', 'EstRowsProcessed', 'EstBytesProcessed',
                'LeafWeightEstRowsWeightedSum', 'LeafWeightEstBytesWeightedSum']

def plan_channel_init(workload):
    plan_channels_init = dict()
    ops = workload.workload_info.all_ops
    ops = np.array([entry.replace(' ', '') for entry in ops])
    ops = np.where(ops == 'NestedLoop', 'NestLoop', ops)
    ops = np.where(ops == 'Materialize', 'Material', ops)
    for c in channels:
        plan_channels_init[c] = dict()
        for node_type in ops:
            plan_channels_init[c][node_type] = 0
    return plan_channels_init

def featurizeDNN(nodes, plan_channels_init):
    if not isinstance(nodes, list):
        nodes = [nodes]
    features = []
    for node in nodes:
        plan_channels = copy.deepcopy(plan_channels_init)
        get_channels_dfs(node, plan_channels)
        needed = torch.from_numpy(get_vecs([plan_channels]))
        node.info['dnn_vec'] = needed
        features.append(needed)
    return features


def cal_by_dfs(root, key, ans):
    if key == 'Plan Rows':
        ans += root._card
    elif key == 'Plan Width':
        ans += root._width
    for plan in root.children:
        cal_by_dfs(plan, key, ans)
    return ans


def get_channels_dfs(root, plan_channels):
    # get node type
    if root.node_type:
        current_node_type = root.node_type
    # EstNodeCost
    plan_channels[channels[0]][current_node_type] += root.cost
    # EstRows
    plan_channels[channels[1]][current_node_type] += root._card
    # EstBytes
    plan_channels[channels[2]][current_node_type] += root._width
    for plan in root.children:
        # EstRowsProcessed
        plan_channels[channels[3]][current_node_type] += plan._card
            # EstBytesProcessed
        plan_channels[channels[4]][current_node_type] += plan._width
        # LeafWeightEstRowsWeightedSum
        plan_channels[channels[5]][current_node_type] += cal_by_dfs(plan, 'Plan Rows', 0)
        # LeafWeightEstBytesWeightedSum
        plan_channels[channels[6]][current_node_type] += cal_by_dfs(plan, 'Plan Width', 0)
    for plan in root.children:
        get_channels_dfs(plan, plan_channels)
    return plan_channels
       
def get_vecs(channels):
    vectors = []
    for plan_channels in channels:
        vector = []
        for channel, values in plan_channels.items():
            tmp_v = [v for v in values.values()]
            vector.extend(tmp_v)
        vectors.append(vector)
    vectors = np.array(vectors)
    return vectors

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size=(128, 64, 32), output_size=2):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.linear4 = nn.Linear(hidden_size[2], output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        # input_size:[batch_size, input_size]
        x = self.linear1(input)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x



class PL_DNN(pl.LightningModule):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.0001):
        super(PL_DNN, self).__init__()
        self.model = model
        self.learning_rate = 0.0001
        self.optimizer_state_dict = optimizer_state_dict

    def forward(self, input):
        return self.model(input)
    
    def diff_normalized(self, p1, p2):
        norm = torch.sum(p1, dim=1)
        pair = (p1 - p2) / norm.unsqueeze(1)
        return pair

    def training_step(self, batch):
        labels = batch['labels']
        vecs1 = batch['vecs1']
        vecs2 = batch['vecs2']
        diff_normalized = self.diff_normalized(vecs1, vecs2).to(torch.float32)
        output = self(diff_normalized)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels.to(torch.long))
        acc = torch.sum(torch.argmax(output, dim=1) == labels).item() / len(labels)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_epoch=True)
        return loss

    def validation_step(self, batch):
        labels = batch['labels']
        vecs1 = batch['vecs1']
        vecs2 = batch['vecs2']
        diff_normalized = self.diff_normalized(vecs1, vecs2).to(torch.float32)
        output = self(diff_normalized)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels.to(torch.long))
        acc = torch.sum(torch.argmax(output, dim=1) == labels).item() / len(labels)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        if self.optimizer_state_dict is not None:
            # Checks the params are the same.
            # 'params': [139581476513104, ...]
            curr = optimizer.state_dict()['param_groups'][0]['params']
            prev = self.optimizer_state_dict['param_groups'][0]['params']
            assert curr == prev, (curr, prev)
            # print('Loading last iter\'s optimizer state.')
            # Prev optimizer state's LR may be stale.
            optimizer.load_state_dict(self.optimizer_state_dict)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            assert optimizer.state_dict(
            )['param_groups'][0]['lr'] == self.learning_rate
            # print('LR', self.learning_rate)
        return optimizer