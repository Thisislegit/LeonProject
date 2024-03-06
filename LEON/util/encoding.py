import sqlparse
import torch

from util import postgres, pg_executor, plans_lib
import models.Treeconv as treeconv




def TreeConvFeaturize(plan_featurizer, subplans, padding_size=200):
    """Returns (featurized plans, tree conv indexes) tensors."""
    assert len(subplans) > 0
    trees, indexes = treeconv.make_and_featurize_trees(subplans,
                                                       plan_featurizer, padding_size)
    return trees, indexes

def PreTreeConvFeaturize(plan_featurizer, subplans, padding_size=200):
    """Returns (featurized plans, tree conv indexes) tensors."""
    assert len(subplans) > 0
    trees, indexes = treeconv.Premake_and_featurize_trees(subplans,
                                                       plan_featurizer, padding_size)
    return trees, indexes

def getencoding_Balsa(sql, hint, workload):
    with pg_executor.Cursor() as cursor:
        node0 = postgres.SqlToPlanNode(sql, comment=hint, verbose=False,
                                       cursor=cursor)[0]
    node = plans_lib.FilterScansOrJoins([node0])[0]
    node.info['sql_str'] = sql
    plans_lib.GatherUnaryFiltersInfo(node)
    postgres.EstimateFilterRows(node)
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    query_vecs = torch.from_numpy(queryFeaturizer(node)).unsqueeze(0)
    if torch.cuda.is_available():
        return [query_vecs.cuda(), node]
    return [query_vecs, node]


if __name__ == '__main__':
    # N.model.train()
    with open('../join-order-benchmark/1a.sql', "r") as f:
        data = f.readlines()
        sql0 = ' '.join(data)
    bais = getCostbais(sql0)
    print(bais)
