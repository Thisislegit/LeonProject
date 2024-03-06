from dataclasses import dataclass, field
import random
from typing import List, Dict
from statistics import mean
from config import read_config
import collections
conf = read_config()
TIME_OUT = 1000000

@dataclass
class EqSetInfo:
    """
    first OVERALL latency of leon
    current OVERALL latency of leon 
    how leon opt this query?
    what query?
    """
    first_latency: float = TIME_OUT
    current_latency: float = TIME_OUT
    opt_time: float = -TIME_OUT
    query_ids: List[str] = field(default_factory=list)
    query_dict: Dict[str, float] = field(default_factory=dict)
    eqset_latency: float = TIME_OUT


class SubplanCost(
        collections.namedtuple(
            'SubplanCost',
            ['subplan', 'cost'],
        )):
    """A collected training data point; wrapper around (subplan, goal, cost).

    Attributes:

      subplan: a balsa.Node.
      goal: a balsa.Node. (deprecated in LEON version)
      cost: the cost of 'goal'.  Specifically: start from subplan, eventually
        reaching 'goal' (joining all leaf nodes & with all filters taken into
        account), what's the cost of the terminal plan?
    """

    # Unused fields: goal
    def ToSubplanGoalHint(self, with_physical_hints=False):
        """subplan's hint_str()--optionally with physical ops--and the goal."""
        return 'subplan=\'{}\''.format(
            self.subplan.hint_str(with_physical_hints),
            ','.join(sorted(self.goal.leaf_ids(alias_only=True))))

    # Unused fields: goal
    def __repr__(self):
        """Basic string representation for quick inspection."""
        return 'SubplanGoalCost(subplan=\'{}\', goal=\'{}\', cost={})'.format(
            self.subplan.hint_str(),
            ','.join(sorted(self.subplan.leaf_ids(alias_only=True))), self.cost)


class Experience:
    def __init__(self, eq_set) -> None:
        self.MinEqNum = int(conf['leon']['MinEqNum'])
        self.MaxEqSets = int(conf['leon']['MaxEqSets'])
        self.LargeTimout = TIME_OUT
        self.__exp = dict() #  map RelId -> [plan1, plan2, ...  ]
        self.__eqSet = dict() # map RelId -> EqSetInfo
        self.__encoding = dict() # map Encoding Id -> (encoding_tuple)
        self.__encoding_index = 0
        for i in eq_set:
            # todo: hand crafted tuned
            self.AddEqSet(i)
            self.GetEqSet()[i].eqset_latency = 2 * TIME_OUT
            self.AppendExp(i, [])

    def AddEncoding(self, encoding_tuple, plan):
        plan.info['index'] = self.__encoding_index
        self.__encoding[self.__encoding_index] = encoding_tuple
        self.__encoding_index += 1
    
    def GetOneEncoding(self, index):
        return self.__encoding[index]
    
    def GetEncoding(self):
        return self.__encoding
    
    def GetEncodingIndex(self):
        return self.__encoding_index

    def OnlyGetExp(self):
        return self.__exp

    def GetQueryId(self, eq: EqSetInfo):
        return self.GetEqSet()[eq].query_ids

    def GetEqSetKeys(self):
        return self.__eqSet.keys()
    
    def GetExpKeys(self):
        return self.__exp.keys()

    def AppendExp(self, eq, plan):
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        if self.haveEq(eq):
            self.GetExp(eq).append(plan)
        else:
            if plan:
                self.__exp[eq] = [plan]
            else:
                self.__exp[eq] = []
    
    def haveEq(self, eq):
        return self.GetExp(eq)

    def IsJoinIdsValid(self, join_ids):
        if join_ids in self.GetEqSet().keys():
            if self.GetEqSet()[join_ids] != self.LargeTimout:
                return True
        return False

    def isCache(self, eq, plan):
        if self.haveEq(eq):
            for curr_plan in self.GetExp(eq):
                if curr_plan.info['sql_str'] == plan.info['sql_str'] and \
                    curr_plan.hint_str() == plan.hint_str():
                    return True
        return False
    
    def ChangeTime(self, eq, plan):
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        for i, curr_plan in enumerate(self.__exp[eq]):
            if curr_plan.info['sql_str'] == plan.info['sql_str'] and \
                curr_plan.hint_str() == plan.hint_str():
                self.__exp[eq][i].info['latency'] = curr_plan.info['latency']
        

    def GetExp(self, eq) -> list:
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        return self.__exp.get(eq)

    def GetEqSet(self) -> dict:
        return self.__eqSet

    def _getEqNum(self):
        return sum(value.opt_time != 0 for value in self.GetEqSet().values())

    def GetPlanNum(self):
        num = 0
        for eq in self.GetEqSetKeys():
            num += len(self.GetExp(eq))
        return num

    def _collectTime(self):
        for eq in self.GetEqSetKeys():
            if self.GetEqSet()[eq].eqset_latency == 2 * TIME_OUT:
                continue
            average = 0
            cnt = 0
            if len(self.GetExp(eq)) > 0:
                for plan in self.GetExp(eq):
                    if plan.info['latency'] != TIME_OUT:
                        cnt += 1
                        average += plan.info['latency']
                if cnt == 0:
                    self.GetEqSet()[eq].eqset_latency = TIME_OUT
                else:
                    self.GetEqSet()[eq].eqset_latency = average / cnt

    def collectRate(self, eq, first_time, tf, query_id):
        """
        Calculate Average Time to rank Eqs.  
        eq: 等价类
        first_time: 第一次leon执行时间
        tf: 当前leon在最终计划中使用该Eq的执行时间
        query_id: 区分查询语句
        """
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        """ Calculate Average Time to rank Eqs.  
        if eq in self.__eqsetTime:
            if query_id in self.__eqsetTime[eq]:
                opt_time = self.__eqsetTime[eq][query_id]
                self.__eqsetTime[eq][query_id] = max(first_time - tf, opt_time)
            else:
                self.__eqsetTime[eq] = dict()
        """

        if eq in self.GetEqSetKeys():
            opt_time = first_time - tf
            query_ids = self.GetEqSet()[eq].query_ids
            query_dict = self.GetEqSet()[eq].query_dict
            eqset_latency = self.GetEqSet()[eq].eqset_latency
            if query_id not in query_ids:
                query_ids.append(query_id)
            if query_id not in query_dict.keys():
                query_dict[query_id] = opt_time
            else:
                query_dict[query_id] = max(opt_time, query_dict[query_id])
            
            self.GetEqSet()[eq] = EqSetInfo(first_latency=first_time,
                                            current_latency=tf,
                                            opt_time=mean(query_dict.values()),
                                            query_ids=query_ids,
                                            query_dict=query_dict,
                                            eqset_latency=eqset_latency)

    def DeleteEqSet(self):
        self._collectTime()
        EqNum = self._getEqNum()
        if EqNum < self.MinEqNum:
            return
        allSet = list(self.GetEqSet().items())
        allSet.sort(key=lambda x: (x[1].eqset_latency, len(x[0])), reverse=True)
        deletenum = min(int(EqNum * 0.15), len(allSet))
        if EqNum - deletenum < self.MinEqNum:
            return
        for i in range(deletenum):
            k, _ = allSet[len(allSet) - 1 - i]
            self.GetEqSet().pop(k)

    def DeleteOneEqset(self, eq):
        del self.GetEqSet()[eq]

    def AddEqSet(self, eq, query_id=None):
        # Limit the Total Number of EqSet
        if self._getEqNum() < self.MaxEqSets: 
            temp = eq.split(',') # sort
            eq = ','.join(sorted(temp))
            if eq not in self.GetEqSetKeys():
                self.GetEqSet()[eq] = EqSetInfo()
                if query_id:
                    self.GetEqSet()[eq].query_ids.append(query_id)
                if not self.GetExp(eq):
                    self.__exp[eq] = []
            else:
                if query_id is not None:
                    if query_id not in self.GetEqSet()[eq].query_ids:
                        self.GetEqSet()[eq].query_ids.append(query_id)
    

    def Getpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.__eqSet.keys():
            for i, j in enumerate(self.GetExp(eq)):
                for k_index in range(i + 1, len(self.GetExp(eq))): 
                    k = self.GetExp(eq)[k_index]   
                    if (j.cost == k.cost):
                        continue
                    if (j.info['sql_str'] == k.info['sql_str']) and \
                        (j.hint_str() == k.hint_str()): 
                        continue
                    if (j.info['sql_str'] != k.info['sql_str']):
                        continue
                    if max(j.info['latency'],k.info['latency']) / \
                        (min(j.info['latency'],k.info['latency']) + 0.001) < 1.05:
                        continue
                    tem = [j, k]
                    pairs.append(tem)
        return pairs

    def PreGetpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.__exp.keys():
            min_dict = dict()
            for j in self.GetExp(eq):
                if j.info['sql_str'] not in min_dict:
                    min_dict[j.info['sql_str']] = j
                else:
                    if j.cost < min_dict[j.info['sql_str']].cost:
                        min_dict[j.info['sql_str']] = j
            for i, j in enumerate(self.GetExp(eq)):
                for k in min_dict.values():
                    if(j.cost == k.cost):
                        continue
                    if (j.info['sql_str'] != k.info['sql_str']):
                        continue
                    if max(j.info['latency'],k.info['latency']) / \
                        (min(j.info['latency'],k.info['latency']) + 0.001) < 1.05:
                        continue
                    tem = [j, k]
                    pairs.append(tem)
                    tem = [k, j]
                    pairs.append(tem)
        return pairs

