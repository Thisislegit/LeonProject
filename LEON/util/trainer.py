import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch.nn.functional as F
import torch.nn as nn
from leon_experience import TIME_OUT

class PL_Leon(pl.LightningModule):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.001):
        super(PL_Leon, self).__init__()
        self.model = model
        self.optimizer_state_dict = optimizer_state_dict
        self.learning_rate = 0.001
        self.eq_summary = dict()
        self.outputs = []

    def forward(self, plans, attns, queryfeature=None):
        if self.model.model_type == 'Transformer':
            if queryfeature is None:
                return self.model(plans, attns)[:, 0]
            return self.model(plans, attns, queryfeature)[:, 0]
        elif self.model.model_type == 'TreeConv':
            return torch.tanh(self.model(queryfeature, plans, attns)).add(1).squeeze(1)

    def getBatchPairsLoss(self, batch):
        """
        batch_pairs: a batch of train pairs
        return. a batch of loss
        """
        labels = batch['labels']
        costs1 = batch['costs1']
        costs2 = batch['costs2']
        encoded_plans1 = batch['encoded_plans1']
        encoded_plans2 = batch['encoded_plans2']
        attns1 = batch['attns1']
        attns2 = batch['attns2']
        queryfeature1 = batch['queryfeature1']
        queryfeature2 = batch['queryfeature2']

        loss_fn = nn.BCELoss()
        # step 1. retrieve encoded_plans and attns from pairs


        # step 2. calculate batch_cali and calied_cost
        # 0是前比后大 1是后比前大
        batsize = costs1.shape[0]
        encoded_plans = torch.cat((encoded_plans1, encoded_plans2), dim=0)
        attns = torch.cat((attns1, attns2), dim=0)
        if queryfeature1 is not None and queryfeature2 is not None:
            queryfeature = torch.cat((queryfeature1, queryfeature2), dim=0)
        else:
            queryfeature = None
        if queryfeature is None:
            cali = self(encoded_plans, attns)
        else:
            cali = self(encoded_plans, attns, queryfeature) 
        costs = torch.cat((costs1, costs2), dim=0)

        calied_cost = torch.log(costs) * cali
        try:
            sigmoid = F.sigmoid(-(calied_cost[:batsize] - calied_cost[batsize:]))
            loss = loss_fn(sigmoid, labels.float()) + 0.2 * torch.abs(calied_cost - torch.log(costs)).mean()
        except:
            print(calied_cost, sigmoid)
        with torch.no_grad():
            prediction = torch.round(sigmoid)
            accuracy = torch.sum(prediction == labels).item() / len(labels)
        
        
        return loss, accuracy


    def training_step(self, batch):
        loss, acc = self.getBatchPairsLoss(batch)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_epoch=True)
        return loss

    def validation_step(self, batch):
        if batch.get('plan_encode') is not None:
            return self.__validation_step_impl(batch)
        loss, acc = self.getBatchPairsLoss(batch)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True)
        return loss
    
    def test_step(self, batch):
        if batch.get('plan_encode') is not None:
            return self.__validation_step_impl(batch)
        loss, acc = self.getBatchPairsLoss(batch)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_epoch=True)
        return loss

    def __validation_step_impl(self, batch):
        def __make_pairs(join_tables, calibrations: torch.tensor, costs, latency, sql):
            assert len(set(join_tables)) == 1

            labels = []
            costs1 = []
            costs2 = [] 
            calibrations1 = []
            calibrations2 = []

            for i in range(len(join_tables)):
                for j in range(i + 1, len(join_tables)):
                    if (costs[i] == costs[j]):
                        continue
                    if (sql[i] != sql[j]) and (latency[i] != TIME_OUT or latency[j] != TIME_OUT):
                        continue
                    if max(latency[i], latency[j]) / min(latency[i], latency[j]) < 1.05:
                        continue
                    if latency[i] > latency[j]:
                        label = 0
                    else:
                        label = 1

                    labels.append(label)
                    costs1.append(costs[i])
                    costs2.append(costs[j])
                    calibrations1.append(calibrations[i])
                    calibrations2.append(calibrations[j])

            # make tensor
            labels = torch.tensor(labels, device=self.device)
            costs1 = torch.tensor(costs1, device=self.device)
            costs2 = torch.tensor(costs2, device=self.device)
            
            try:
                calibrations1 = torch.stack(calibrations1)
                calibrations2 = torch.stack(calibrations2)
            except:
                calibrations1 = None
                calibrations2 = None
            return labels, costs1, costs2, calibrations1, calibrations2

        plans = batch['plan_encode']
        attns = batch['att_encode']
        costs = batch['cost']
        labels = batch['latency']
        join_tables = batch['join_tables']
        sql = batch['sql']
        queryfeature = batch['queryfeature']
        eq = ','.join(sorted(join_tables[0].split(' ')))

        calibrations = self(plans, attns, queryfeature)
        labels, costs1, costs2, calibrations1, calibrations2 = __make_pairs(join_tables, calibrations, costs, labels, sql)

        if calibrations1 is None or calibrations2 is None:
            self.eq_summary[eq] = (0, 0) # 准确率 pair数
            return None, None, None
        
        loss_fn = nn.BCELoss()

        calied_cost_1 = torch.log(costs1) * calibrations1
        calied_cost_2 = torch.log(costs2) * calibrations2

        sigmoid = F.sigmoid(-(calied_cost_1 - calied_cost_2))
        loss = loss_fn(sigmoid, labels.double())

        prediction = torch.round(sigmoid)
        accuracy = torch.sum(prediction == labels).item() / len(labels)

        self.eq_summary[eq] = (round(accuracy, 3), len(labels))
        self.outputs.append((accuracy, loss, len(labels)))
        return accuracy, loss, len(labels)

    def on_test_epoch_end(self):
        accuracy = 0
        total = 0
        loss = 0
        for output in self.outputs:
            if output is not None:
                accuracy += output[0] * output[2]
                loss += output[1] * output[2]
                total += output[2]
        if total == 0:
            return
        
        accuracy /= total
        loss /= total
        self.log_dict({'test_acc': accuracy, 'test_loss': loss}, on_epoch=True)

        # clear outputs
        self.outputs.clear()
        return accuracy
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
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

