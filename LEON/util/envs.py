"""Workload definitions."""
import glob
import os
import numpy as np
import pickle
from . import hyperparams, plans_lib, postgres, encoding
from models import Transformer

import random
import re
import torch
import sys
sys.path.append('..')

_EPSILON = 1e-6


def load_training_query(query_path):
    train_queries = []
    with open(query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            train_queries.append((arr[0], arr[1]))
    print("Read", len(train_queries), "test queries.")
    return train_queries

def load_sql(file_list: list, training_query=None):
    """
    :param file_list: list of query file in str
    :return: list of sql query string
    """
    sqls = []
    for file_str in file_list:
        if training_query:
            sqls.append(training_query[file_str])
        else:
            sqlFile = './my_job/' + file_str + '.sql'
            if not os.path.exists(sqlFile):
                raise IOError("File Not Exists!")
            with open(sqlFile, 'r') as f:
                data = f.read().splitlines()
                sql = ' '.join(data)
            sqls.append(sql)
            f.close()
    return sqls

def PlanToNode(workload, plans, sql=None):
    """
    input. plans 一个 message 等价类包括多条 plans
    output. nodes
        sql 信息在 node.info['sql_str'], cost 信息在 node.cost, hint 信息在 node.hint_str(), join_tables 信息在 node.info['join_tables']
    """
    nodes = []
    for i in range(len(plans)):
        # print("plans[{}]\n".format(i))
        node = postgres.ParsePostgresPlanJson_1(plans[i], workload.workload_info.alias_to_names)
        if i == 0:
            if node.info['join_cond'] == ['']:
                return None
            if sql is None:
                temp = node.to_sql(node.info['join_cond'], with_select_exprs=True)
            else:
                temp = sql
            node.info['sql_str'] = temp
        node.info['sql_str'] = temp
        nodes.append(node)
    
    return nodes

def ParseSqlToNode(path):
    base = os.path.basename(path)
    query_name = os.path.splitext(base)[0]
    with open(path, 'r') as f:
        sql_string = f.read()
    node, json_dict = postgres.SqlToPlanNode(sql_string)
    # print(node)
    node.info['path'] = path
    node.info['sql_str'] = sql_string
    node.info['query_name'] = query_name
    node.info['explain_json'] = json_dict
    node.GetOrParseSql()
    return node


class Workload(object):

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('query_dir', None, 'Directory to workload queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern.  Otherwise, use all queries.' \
            '  Example: 29*.sql.'
        )
        p.Define(
            'loop_through_queries', False,
            'Loop through a random permutation of queries? '
            'Desirable for evaluation.')
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treating all queries' \
            ' as training nodes.'
        )
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Join operators to learn.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Scan operators to learn.')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Subclasses should populate these fields.
        self.query_nodes = None
        self.workload_info = None
        self.train_nodes = None
        self.test_nodes = None

        if p.loop_through_queries:
            self.queries_permuted = False
            self.queries_ptr = 0

    def _ensure_queries_permuted(self, rng):
        """Permutes queries once."""
        if not self.queries_permuted:
            self.query_nodes = rng.permutation(self.query_nodes)
            self.queries_permuted = True

    def _get_sql_set(self, query_dir, query_glob):
        if query_glob is None:
            return set()
        else:
            globs = query_glob
            if type(query_glob) is str:
                globs = [query_glob]
            sql_files = np.concatenate([
                glob.glob('{}/{}'.format(query_dir, pattern))
                for pattern in globs
            ]).ravel()
        sql_files = set(sql_files)
        return sql_files

    def Queries(self, split='all'):
        """Returns all queries as balsa.Node objects."""
        assert split in ['all', 'train', 'test'], split
        if split == 'all':
            return self.query_nodes
        elif split == 'train':
            return self.train_nodes
        elif split == 'test':
            return self.test_nodes

    def WithQueries(self, query_nodes):
        """Replaces this Workload's queries with 'query_nodes'."""
        self.query_nodes = query_nodes
        self.workload_info = plans_lib.WorkloadInfo(query_nodes)

    def FilterQueries(self, query_dir, query_glob, test_query_glob):
        all_sql_set_new = self._get_sql_set(query_dir, query_glob)
        test_sql_set_new = self._get_sql_set(query_dir, test_query_glob)
        assert test_sql_set_new.issubset(all_sql_set_new), (test_sql_set_new,
                                                            all_sql_set_new)

        all_sql_set = set([n.info['path'] for n in self.query_nodes])
        assert all_sql_set_new.issubset(all_sql_set), (
            'Missing nodes in init_experience; '
            'To fix: remove data/initial_policy_data.pkl, or see README.')

        query_nodes_new = [
            n for n in self.query_nodes if n.info['path'] in all_sql_set_new
        ]
        train_nodes_new = [
            n for n in query_nodes_new
            if test_query_glob is None or n.info['path'] not in test_sql_set_new
        ]
        test_nodes_new = [
            n for n in query_nodes_new if n.info['path'] in test_sql_set_new
        ]
        assert len(train_nodes_new) > 0

        self.query_nodes = query_nodes_new
        self.train_nodes = train_nodes_new
        self.test_nodes = test_nodes_new

    def UseDialectSql(self, p):
        dialect_sql_dir = p.engine_dialect_query_dir
        for node in self.query_nodes:
            assert 'sql_str' in node.info and 'query_name' in node.info
            path = os.path.join(dialect_sql_dir,
                                node.info['query_name'] + '.sql')
            assert os.path.isfile(path), '{} does not exist'.format(path)
            with open(path, 'r') as f:
                dialect_sql_string = f.read()
            node.info['sql_str'] = dialect_sql_string


class JoinOrderBenchmark(Workload):

    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        module_dir = os.path.abspath(os.path.dirname(__file__) + '/../')
        p.query_dir = os.path.join(module_dir, 'workloads/job/')
        if not os.path.exists(p.query_dir):
            raise IOError('File Not Exists!')
        return p

    def __init__(self, params):
        super().__init__(params)
        p = params
        self.query_nodes, self.train_nodes, self.test_nodes = \
            self._LoadQueries()
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        self.workload_info.SetPhysicalOps(p.search_space_join_ops,
                                          p.search_space_scan_ops)

    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        assert test_sql_set.issubset(all_sql_set)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set)
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]

        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes

class JoinOrderBenchmark_Train(JoinOrderBenchmark):
    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        # module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        #  p.query_dir = os.path.join('/home/ht/PycharmProjects/pythonProject3', 'join-order-benchmark')
        module_dir = os.path.abspath(os.path.dirname(__file__)) + '/../'    
        print(module_dir)
        p.query_dir = os.path.join(module_dir + './workloads/training_query/job_train.txt')
        if not os.path.exists(p.query_dir):
            raise IOError('File Not Exists!')
        return p
    
    def __init__(self, params):
        super().__init__(params)

    def ParseSqlToNode(self, sqlfile, sql_string):
        query_name = sqlfile
        node, json_dict = postgres.SqlToPlanNode(sql_string)
        node.info['sql_str'] = sql_string
        node.info['query_name'] = query_name
        node.info['explain_json'] = json_dict
        node.GetOrParseSql()
        return node

    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set : list = load_training_query(p.query_dir)
        test_sql_set : list = self._get_sql_set(p.query_dir, p.test_query_glob)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set, key=lambda x: x[0])
        all_nodes = [self.ParseSqlToNode(sqlfile, sql) for sqlfile, sql in all_sql_list]

        train_nodes = all_nodes
        test_nodes = []
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes


class CEBbenchmark(Workload):

    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        # module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        #  p.query_dir = os.path.join('/home/ht/PycharmProjects/pythonProject3', 'join-order-benchmark')
        p.query_dir = 'CEB_benchmark'
        if not os.path.exists(p.query_dir):
            raise IOError('File Not Exists!')
        return p

    def __init__(self, params):
        super().__init__(params)
        p = params
        self.query_nodes, self.train_nodes, self.test_nodes = \
            self._LoadQueries()
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        self.workload_info.SetPhysicalOps(p.search_space_join_ops,
                                          p.search_space_scan_ops)

    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        assert test_sql_set.issubset(all_sql_set)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set)
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]

        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes


class TPCHbenchmark(Workload):

    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        # module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        #  p.query_dir = os.path.join('/home/ht/PycharmProjects/pythonProject3', 'join-order-benchmark')
        p.query_dir = 'tpch_query'
        if not os.path.exists(p.query_dir):
            raise IOError('File Not Exists!')
        return p

    def __init__(self, params):
        super().__init__(params)
        p = params
        self.query_nodes, self.train_nodes, self.test_nodes = \
            self._LoadQueries()
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        self.workload_info.SetPhysicalOps(p.search_space_join_ops,
                                          p.search_space_scan_ops)

    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        assert test_sql_set.issubset(all_sql_set)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set)
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]

        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes


class RunningStats(object):
    """Computes running mean and standard deviation.

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs.Record(np.random.randn())
        print(rs.Mean(), rs.Std())
    """

    def __init__(self, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def Record(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def Mean(self):
        return self.m if self.n else 0.0

    def Variance(self):
        return self.s / (self.n) if self.n else 0.0

    def Std(self, epsilon_guard=True):
        eps = 1e-6
        std = np.sqrt(self.Variance())
        if epsilon_guard:
            return np.maximum(eps, std)
        return std

def wordload_init(workload_type):
    path = f'./log/workload_{workload_type}.pkl'
    
    if not os.path.exists(path):
        if workload_type == 'job_training':
            workload = JoinOrderBenchmark_Train(JoinOrderBenchmark_Train.Params())
        else:
            workload = JoinOrderBenchmark(JoinOrderBenchmark.Params())
        workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)
        workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(workload.workload_info.rel_ids)
        # dump queryFeaturizer and workload
        with open(path, 'wb') as f:
            pickle.dump(workload, f)
    else:
        with open(path, 'rb') as f:
            workload = pickle.load(f)
    print("Read Workload:", workload_type)
    return workload

def CurrCache(curr_exec, plan):
    for curr_plan in curr_exec:
        if curr_plan[0].info['sql_str'] == plan.info['sql_str'] and curr_plan[0].hint_str() == plan.hint_str():
            return True
    return False

def load_train_files(workload_type):
    if workload_type == 'job_training':
        training_query = load_training_query("./workloads/training_query/job_train.txt")
        train_files = [i[0] for i in training_query]
        training_query = [i[1] for i in training_query]
    else:
        train_files = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a',
                    '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', 
                    '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', 
                    '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', 
                    '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c',
                    '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a',
                    '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b',
                    '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', 
                    '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
                    '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c']
        random.shuffle(train_files)
        train_files = train_files * 3
        training_query = load_sql(train_files)
        
    return train_files, training_query

def find_alias(training_query):
    a = []
    for sql_query in training_query:
        # 提取FROM和WHERE之间的内容
        from_where_content = re.search('FROM(.*)WHERE', sql_query.replace("\n", "")).group(1)

        # 提取别名
        aliases = re.findall(r'AS (\w+)', from_where_content)
        
        aliases = ",".join(sorted(aliases))
        a.append(aliases)

    return a

def plans_encoding(plans, configs, op_name_to_one_hot, plan_parameters, feature_statistics):
    '''
    input. a list of plans in type of json
    output. (seq_encoding, run_times, attention_mask, loss_mask)
        - seq_encoding torch.Size([1, 760])
    '''
    seqs = []
    attns = []
    for x in plans:        
        seq_encoding, run_times, attention_mask, loss_mask, database_id = Transformer.get_plan_encoding(
            x, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        ) # run_times 无法获取
        seqs.append(seq_encoding) 
        attns.append(attention_mask)

    seqs = torch.stack(seqs, dim=0)
    attns = torch.stack(attns, dim=0)

    return seqs, run_times, attns, loss_mask

def leon_encoding(model_type, X, require_nodes=False, workload=None,
                  queryFeaturizer=None, nodeFeaturizer=None, 
                  configs=None, op_name_to_one_hot=None, 
                  plan_parameters=None, feature_statistics=None, sql=None):
    if model_type == "TreeConv" or require_nodes:
        nodes = PlanToNode(workload, X, sql)
        if nodes is None:
            return None, None, None, None
        plans_lib.GatherUnaryFiltersInfo(nodes)
        postgres.EstimateFilterRows(nodes) 
        null_nodes = plans_lib.Binarize(nodes)
        query_vecs = torch.from_numpy(queryFeaturizer(nodes[0]))
        OneQueryFeature = query_vecs.unsqueeze(0)
        
        for node in nodes:
            node.info['query_feature'] = query_vecs
    if model_type == "Transformer":
        if require_nodes == False:
            OneNode = PlanToNode(workload, [X[0]])[0]
            plans_lib.GatherUnaryFiltersInfo(OneNode)
            postgres.EstimateFilterRows(OneNode)  
            OneQueryFeature = queryFeaturizer(OneNode)
            OneQueryFeature = torch.from_numpy(OneQueryFeature).unsqueeze(0)
        encoded_plans, _, attns, _ = plans_encoding(X, configs, op_name_to_one_hot, plan_parameters, feature_statistics) # encoding. plan -> plan_encoding / seqs torch.Size([26, 1, 760])
        queryfeature = OneQueryFeature.repeat(encoded_plans.shape[0], 1)
        if require_nodes:
            return encoded_plans, attns, queryfeature, nodes
        else:
            return encoded_plans, attns, queryfeature, None
    elif model_type == "TreeConv":
        trees = []
        indexes = []
        for node in null_nodes:
            tree, index = encoding.PreTreeConvFeaturize(nodeFeaturizer, [node]) 
            trees.append(tree)
            indexes.append(index)
        queryfeature = OneQueryFeature.repeat(len(trees), 1)
        if require_nodes:
            return trees, indexes, queryfeature, nodes
        else:
            return trees, indexes, queryfeature, None
