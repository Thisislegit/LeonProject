from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch.nn.functional as F
import os
from ray.util import ActorPool
from util import postgres
from util.pg_executor import ActorThatQueries, actor_call_leon
from util import envs
from util.envs import load_sql
from util import plans_lib
import pytorch_lightning.loggers as pl_loggers
import pickle
import math
from util.dataset import prepare_dataset
from leon_experience import Experience, TIME_OUT
from util.trainer import PL_Leon
import numpy as np
import ray
import time
from config import read_config
import models.Treeconv as treeconv
from models.Transformer import *
from models.DNN import *
import gc
from collections import namedtuple

ExecPlan = namedtuple('ExecPlan', 
                      ['plan', 'timeout', 'eq_set', 'cost'])

DEVICE = 'cpu'
conf = read_config()
model_type = conf['leon']['model_type']


def load_model(model_path: str, queryfeaturizer, nodefeaturizer, ckpt=False):
    if not os.path.exists(model_path):
        if model_type == "Transformer":
            print("load transformer model")
            model = SeqFormer(
                            input_dim=configs['node_length'],
                            hidden_dim=128,
                            output_dim=1,
                            mlp_activation="ReLU",
                            transformer_activation="gelu",
                            mlp_dropout=0.2,
                            transformer_dropout=0.2,
                            query_dim=configs['query_dim'],
                            padding_size=configs['pad_length']
                        ).to(DEVICE)
        elif model_type == "TreeConv":
            print("load treeconv model")
            # model = treeconv.TreeConvolution(666, 50, 1).to(DEVICE)
            model = treeconv.ResNet(
                queryfeaturizer.Dim(), 
                nodefeaturizer.Dim(), 
                1, 
                treeconv.ResidualBlock, 
                [1, 1, 1, 1]).to(DEVICE)
            print(f"Model size: ({queryfeaturizer.Dim()}, {nodefeaturizer.Dim()})")
        dnn_model = DNN(len(list(plan_channels_init.values())[0].values()) * len(plan_channels_init.keys()), [512, 256, 128], 2)
        print(f"DNN Model size: ({len(list(plan_channels_init.values())[0].values()) * len(plan_channels_init.keys())}, [512, 256, 128], 2)")
        torch.save(dnn_model, "./log/dnn_model.pth")
        dnn_model = PL_DNN(dnn_model)
        torch.save(model, model_path)
        model = PL_Leon(model, prev_optimizer_state_dict)
    else:
        assert ckpt == True
        dnn_model = torch.load("./models/dnn.pth", map_location=DEVICE).to(DEVICE)
        dnn_model = PL_DNN(dnn_model)
        model = torch.load("./models/resnet.pth", map_location=DEVICE).to(DEVICE)
        model = PL_Leon(model, prev_optimizer_state_dict)
    
    return model, dnn_model

def getPG_latency(query, hint=None, ENABLE_LEON=False, timeout_limit=None, curr_file=None):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency_sum = 0
    if timeout_limit is None:
        timeout_limit = TIME_OUT # TIME_OUT
    cnt = 1
    for c in range(cnt):
        latency, json_dict = postgres.GetLatencyFromPg(
            query, 
            hint, 
            ENABLE_LEON, 
            verbose=False, 
            check_hint_used=False, 
            timeout=timeout_limit, 
            dropbuffer=False,
            curr_file=curr_file)
        latency_sum = latency_sum + latency
    pg_latency = latency_sum / cnt
    if pg_latency == timeout_limit:
        pg_latency = TIME_OUT
    if ENABLE_LEON and json_dict == []:
        json_dict = postgres.getPlans(
            query, 
            hint, 
            check_hint_used=False, 
            ENABLE_LEON=ENABLE_LEON, 
            curr_file=curr_file)[0][0][0]
        
    return pg_latency, json_dict

def get_calibrations(model, seqs, attns, queryfeature):
    seqs = seqs.to(DEVICE)
    attns = attns.to(DEVICE)
    queryfeature = queryfeature.to(DEVICE)
    cost_iter = 10 # Get calibration for cost_iter times
    model.model.eval()
    with torch.no_grad():
        for i in range(cost_iter):
            if model_type == "Transformer":
                cali = model.model(seqs, attns, queryfeature)[:, 0] 
            elif model_type == "TreeConv":
                cali = torch.tanh(
                    model.model(seqs, attns, queryfeature)).add(1).squeeze(1)
            if i == 0:
                cali_all = cali.unsqueeze(1)
            else:
                cali_all = torch.cat((cali_all, cali.unsqueeze(1)), dim=1)
    return cali_all

def get_ucb_idx(cali_all, pgcosts):    
    cali_mean = cali_all.mean(dim = 1) 
    cali_var = cali_all.var(dim = 1)
    # costs = pgcosts 
    cost_t = torch.mul(cali_mean, costs)
    # cali_min, _ = cost_t.min(dim = 0) 
    # ucb = cali_var / cali_var.max() - cali_min / cali_min.max()
    # ucb = cali_var / cali_var.max()
    ucb = cost_t
    # ucb = cali_var / cali_var.max() - cost_t / cost_t.max() # [# of plan]
    ucb_sort_idx = torch.argsort(ucb, descending=True)
    ucb_sort_idx = ucb_sort_idx.tolist()

    return ucb_sort_idx

def getNodesCost(nodes):
    pgcosts = [] # 存所有 plan 的 pg cost
    for node in nodes:
        pgcost = node.cost
        pgcost_log = math.log(pgcost)
        pgcosts.append(pgcost_log)
    
    return pgcosts

def initEqSet():
    train_file, training_query = envs.load_train_files(conf['leon']['workload_type'])
    equ_tem = envs.find_alias(training_query)
    equ_set = set() 
    for i in equ_tem:
        e_tem = i.split(',')
        e_tem = ','.join(sorted(e_tem))
        equ_set.add(e_tem)

    return equ_set

def collects(finnode: plans_lib.Node, actor, exp: Experience, query_id, model, execution_time):
    """
    Collects join IDs from the given `finnode` and performs degradation analysis.

    Args:
        finnode (plans_lib.Node): The final node of the query plan.
        actor: The ray actor object (for c).
        exp (Experience): The experience object.
        timeout: The timeout value.
        currTotalLatency: The current total latency.
        sql: The SQL query.
        query_id: The query ID.
        model: The model object.

    Returns:
        None
    """
    join_ids_to_append = []
    allPlans = [finnode]
    while (allPlans):
        currentNode = allPlans.pop(0)
        allPlans.extend(currentNode.children)
        if currentNode.IsJoin():
            cur_join_ids = ','.join(
                sorted([i.split(" AS ")[-1] for i in currentNode.leaf_ids()]))
            join_ids_to_append.append(cur_join_ids)
    
    for join_id in join_ids_to_append:
        exp_key = Exp.GetExpKeys()
        temp = join_id.split(',') # sort
        if len(temp) <= 3 or execution_time < 3000:
            return
        join_id = ','.join(sorted(temp))
        if join_id not in exp_key:
            print('degradation collect:', join_id)
            exp.AddEqSet(join_id, query_id)
            ray.get(actor.reload_model.remote(exp.GetEqSetKeys(), model.eq_summary))
            return

def load_callbacks(logger):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=2,
        min_delta=0.001,
        check_on_train_epoch_end=False
    ))
    if logger:
        callbacks.append(plc.ModelCheckpoint(
            dirpath= logger.experiment.dir,
            monitor='val_scan',
            filename='best-{epoch:02d}-{val_scan:.3f}',
            save_top_k=1,
            mode='min',
            save_last=False
        ))

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))
    return callbacks

def _batch(trees, indexes, padding_size=200):
    """
    Pad the input trees and indexes to a specified padding size.

    Args:
        trees (List[torch.Tensor]): List of input trees.
        indexes (List[torch.Tensor]): List of input indexes.
        padding_size (int, optional): The desired padding size. Defaults to 200.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded trees and indexes.
    """
    # 获取 batchsize
    batch_size = len(trees)
    tree_embedding_size = trees[0].size(1)
    # 初始化填充后的张量
    padded_trees = torch.zeros((batch_size, tree_embedding_size, padding_size))
    padded_indexes = torch.zeros((batch_size, padding_size, 1))

    for i in range(batch_size):
        # 获取当前样本的原始树和索引张量
        tree = trees[i]
        index = indexes[i]

        # 计算需要填充的列数
        padding_cols_tree = max(0, padding_size - tree.size(2))
        padding_cols_index = max(0, padding_size - index.size(1))

        # 使用 F.pad 进行填充
        padded_tree = F.pad(tree, (0, padding_cols_tree), value=0)
        padded_index = F.pad(index, (0, 0, 0 , padding_cols_index), value=0)

        # 将填充后的张量放入结果中
        padded_trees[i, :, :] = padded_tree
        padded_indexes[i, :, :] = padded_index

    return padded_trees, padded_indexes.long()
    
def PlanToExecute(plan, temp, exp: Experience, encoded_plan, attn):
    """
    performing necessary operations and before the execution plan.
    """
    global q_recieved_cnt
    needed = featurizeDNN(plan, plan_channels_init)[0]
    exp.AddEncoding(
        (encoded_plan, attn, needed), plan)
    exec_plan.append(
        ExecPlan(
            plan=plan,
            timeout=(pg_time1[q_recieved_cnt] * TIME_OUT_Ratio + planning_time),
            eq_set=temp,
            cost=plan.cost
        )
    )

def ConsecutivePlanning(sql, filename):
    postgres.getPlans(sql, 
                      None, 
                      check_hint_used=False, 
                      ENABLE_LEON=True, 
                      curr_file=filename)

def FindNodeInExp(Nodes: list, Exp: Experience):
    """
    Find the node in the experience, We use the join tables to find the node.
    Note that we include child nodes in the search.

    Args:
        node: The node to find.
        exp: The experience.

    Returns:
        The node if found, None otherwise.
    """
    leon_node = []
    for node in Nodes:
        all_node = []
        all_plan = [node]
        while (all_plan):
            currentNode = all_plan.pop(0)
            all_plan.extend(currentNode.children)
            if currentNode.IsJoin():
                all_node.append(currentNode)
        for child_node in all_node:
            tbls = [table.split(' ')[-1] for table in child_node.leaf_ids(with_alias=True)]
            eq_temp = ','.join(sorted(tbls))
            if eq_temp in Exp.GetEqSet():
                print(','.join(sorted(tbls)))
                child_node.info['join_tables'] = ','.join(sorted(tbls))
                leon_node.append(child_node)
    return leon_node


def Subsitution(leon_nodes, all_nodes, Exp: Experience):
    """
    Substitutes execution time for nodes in the experience.

    Args:
        leon_nodes (list of plans_lib.Node): List of nodes from LEON.
        all_nodes (list of plans_lib.Node): List of all nodes.
        Exp (Experience): The experience object.

    Returns:
        None
    """
    for node2 in leon_nodes:
        for i, node1 in enumerate(all_nodes):
            temp = node1.info['join_tables'].split(',') # sort
            temp = ','.join(sorted(temp))
            if temp == node2.info['join_tables']:
                # If the node is found, 
                # we will substitute the execution time
                if round(node1.cost * 100) / 100.0 == node2.cost:
                    Exp.collectRate(
                        node2.info['join_tables'], 
                        first_time[curr_file[q_recieved_cnt]], 
                        tf_time[q_recieved_cnt], 
                        curr_file[q_recieved_cnt])
                    c_plan = node1
                    if node2.actual_time_ms is not None:
                        c_plan.info['latency'] = node2.actual_time_ms
                    else:
                        if c_plan.info.get('latency') is None:
                            if not Exp.isCache(c_plan.info['join_tables'], c_plan) and \
                                not envs.CurrCache(exec_plan, c_plan):
                                PlanToExecute(
                                    c_plan, temp, Exp, encoded_plans[i], attns[i])
                            break
                    
                    # Find the node in the experience,
                    # We use the join tables to find the node.
                    # If the node is found, we will substitute the execution time
                    # If the node is not found, 
                    # we will add the node to the experience
                    if not Exp.isCache(c_plan.info['join_tables'], c_plan):
                        needed = featurizeDNN(c_plan, plan_channels_init)[0]
                        Exp.AddEncoding(
                            (encoded_plans[i], attns[i], needed), c_plan)
                        Exp.AppendExp(c_plan.info['join_tables'], c_plan)
                    else:
                        Exp.ChangeTime(c_plan.info['join_tables'], c_plan)
                    break
            else:
                break

if __name__ == '__main__':
    file_path = ["./log/messages.pkl", './log/model.pth', './log/dnn_model.pth']
    for file_path1 in file_path:
        if os.path.exists(file_path1):
            os.remove(file_path1)
            print(f"File {file_path1} has been successfully deleted.")
        else:
            print(f"File {file_path1} does not exist.")
    
    pretrain = False
    if pretrain:
        checkpoint = torch.load("./log/SimModel3.pth", map_location=DEVICE)
        torch.save(checkpoint, "./log/model.pth")
        print("load SimModel success")
    if not os.path.exists("./log/exp_v5.pkl"):
        exp_cache = dict()
    else:
        with open("./log/exp_v5.pkl", "rb") as file:
            exp_cache = pickle.load(file)

    # Ray Config
    with open ("./conf/namespace.txt", "r") as file:
        namespace = file.read().replace('\n', '')
    with open ("./conf/ray_address.txt", "r") as file:
        ray_address = file.read().replace('\n', '')
    context = ray.init(address=ray_address, namespace=namespace,
                        _temp_dir=conf['leon']['ray_path'] + "/log/ray") 
    print(context.address_info)
    # We use several ray actor to execute the queries in parallel
    our_ports = eval(conf['leon']['other_leon_port'])
    our_ports.append(int(conf['leon']['Port']))
    ports = eval(conf['leon']['other_db_port'])
    ports.append(int(conf['PostgreSQL']['port']))
    actors = [ActorThatQueries.options(name=f"actor{port}").remote(port, our_port) \
               for port, our_port in zip(ports, our_ports)]
    pool = ActorPool(actors)
    
    # Load Train queries
    train_files, training_query = envs.load_train_files(conf['leon']['workload_type'])
    chunk_size = 6 # the # of sqls in a chunk
    min_batch_size = 256
    TIME_OUT_Ratio = 2
    model_path = "./log/model.pth" 
    message_path = "./log/messages.pkl"
    prev_optimizer_state_dict = None
    dnn_prev_optimizer_state_dict = None
    # Training experience
    Exp = Experience(eq_set=initEqSet())
    eqset = Exp.GetEqSet()
    
    print("Init workload and equal set keys")
    workload = envs.wordload_init(conf['leon']['workload_type'])

    # Init ML Model Configs
    plan_channels_init = plan_channel_init(workload=workload)
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    if model_type == "Transformer":
        statistics_file_path = "./statistics.json"
        feature_statistics = load_json(statistics_file_path)
        add_numerical_scalers(feature_statistics)
        op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    elif model_type == "TreeConv":
        nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(workload.workload_info)
    model, dnn_model = load_model(model_path, queryFeaturizer, nodeFeaturizer)
    model.eq_summary = {key: 0 for key in eqset}
    first_time = dict()
    last_train_pair = 0
    max_query_latency1 = 0
    logger =  pl_loggers.WandbLogger(
        save_dir=os.getcwd() + '/logs', 
        name=f"LEON_{conf['leon']['workload_type']}", 
        project=conf['leon']['wandb_project'])
    for key in conf:
        logger.log_hyperparams(conf[key])
    my_step = 0
    same_actor = ray.get_actor('leon_server')
    task_counter = ray.get_actor('counter')
    runtime_pg = 0
    runtime_leon = 0
    min_exec_num = 2
    max_exec_num = 30
    train_gpu = int(conf['leon']['train_gpu'])
    remote = bool(conf['leon']['remote'])
    pct = float(conf['leon']['pct']) # 执行 percent 比例的 plan
    planning_time = 10000 # pg timout会考虑planning时间

    # ===== ITERATION OF CHUNKS ====
    ch_start_idx = 0 
    # The start idx of the current chunk in train_files
    while ch_start_idx + chunk_size <= len(train_files):
        print(f"\n+++++++++ a chunk of sql from {ch_start_idx}  ++++++++")
        sqls_chunk = load_sql(list(range(ch_start_idx, ch_start_idx + chunk_size)), training_query=training_query)
        curr_file = train_files[ch_start_idx : ch_start_idx + chunk_size]
        print(train_files[ch_start_idx : ch_start_idx + chunk_size])
        time_ratio = []
        tf_time = []
        pg_time1 = []
        Nodes = []
        # ++++ PHASE 1. ++++ send a chunk of queries with ENABLE_LEON=True
        for q_send_cnt in range(chunk_size):
            print(f"------------- sending query {q_send_cnt} starting from idx {ch_start_idx} ------------")
            # Get expert optimizer latency
            query_latency1, _ = getPG_latency(
                sqls_chunk[q_send_cnt], ENABLE_LEON=False, timeout_limit=0)
            print("latency pg ", query_latency1)
            # Get leon optimizer latency, explain the plan first
            # Since we make sure the eqset is consacutive
            # So we explain the query without actual execution
            ConsecutivePlanning(sqls_chunk[q_send_cnt], curr_file[q_send_cnt])
            query_latency2, json_dict = getPG_latency(
                sqls_chunk[q_send_cnt], 
                ENABLE_LEON=True, 
                timeout_limit=int(conf['leon']['leon_timeout']), 
                curr_file=curr_file[q_send_cnt])
            print("latency leon ", query_latency2)
            node = postgres.ParsePostgresPlanJson(json_dict)
            max_query_latency1 = max(max_query_latency1, query_latency1)
            Nodes.append(node)
            time_ratio.append(query_latency2 / query_latency1)
            tf_time.append(query_latency2)
            pg_time1.append(query_latency1)
            logger.log_metrics(
                {f"Query/{curr_file[q_send_cnt]}pg_latency": query_latency1}, step=my_step)
            logger.log_metrics(
                {f"Query/{curr_file[q_send_cnt]}leon_latency": query_latency2}, step=my_step)
            if curr_file[q_send_cnt] not in first_time:
                first_time[curr_file[q_send_cnt]] = query_latency1 # 第一次pgtime
        
        logger.log_metrics({f"Runtime/pg_latency": sum(pg_time1)}, step=my_step)
        logger.log_metrics({f"Runtime/leon_latency": sum(tf_time)}, step=my_step)
        runtime_pg += sum(pg_time1)
        runtime_leon += sum(tf_time)
        logger.log_metrics({f"Runtime/all_pg": runtime_pg}, step=my_step)
        logger.log_metrics({f"Runtime/all_leon": runtime_leon}, step=my_step)
        
        ############### Collect New Eq Sets ##############################  
        for q_send_cnt in range(chunk_size):
            curNode = Nodes[q_send_cnt]
            if curNode:
                collects(
                    curNode, task_counter, Exp, curr_file[q_send_cnt], model, tf_time[q_send_cnt]
                )
        for q_send_cnt in range(chunk_size):
            postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
            ray.get(task_counter.WriteOnline.remote(True))
            postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
            ray.get(task_counter.WriteOnline.remote(False)) 

        ##########################################################
        # If the newly executed node is from the same eqset,
        # we will not execute it again
        leon_node = FindNodeInExp(Nodes, Exp)

        # ==== ITERATION OF RECIEVED QUERIES IN A CHUNK ====
        q_recieved_cnt = 0 # the # of recieved queries; value in [1, chunk_size]
        curr_QueryId = "" # to imply the difference of QueryId between two messages
        # To ensure that all sent sqls are processed as messages before loading .pkl file
        completed_tasks = ray.get(same_actor.GetCompletedTasks.remote())
        recieved_task = ray.get(task_counter.GetRecievedTask.remote())
        if completed_tasks == recieved_task:
            PKL_READY = True
        else:
            PKL_READY = False
        while (not PKL_READY):
            time.sleep(0.1)
            print("waiting for PKL_READY ...")
            completed_tasks = ray.get(same_actor.GetCompletedTasks.remote())
            recieved_task = ray.get(task_counter.GetRecievedTask.remote())
            if completed_tasks == recieved_task:
                PKL_READY = True
            else:
                PKL_READY = False
        exec_plan = []
        start_time = time.time()
        # ++++ PHASE 2. ++++ get messages of a chunk, nodes, and experience
        PKL_exist = True
        if os.path.exists(message_path):
            with open(message_path, "rb") as file:
                while(PKL_exist):
                    try:
                        message = pickle.load(file)
                    except:
                        PKL_exist = False # the last message is already loaded
                        break
                    curr_sql_id = message[0]['QueryId']
                    q_recieved_cnt = curr_file.index(curr_sql_id) # start to recieve equal sets from a new sql
                    if curr_QueryId != message[0]['QueryId']: # the QueryId of the first plan in the message
                        print(f"------------- recieving query {q_recieved_cnt} starting from idx {ch_start_idx} ------------")
                        curr_QueryId = message[0]['QueryId']
                    print(f">>> message with {len(message)} plans")
                    
                    # STEP 1) get node
                    if model_type == "Transformer":
                        encoded_plans,\
                        attns,\
                        queryfeature, \
                        nodes = envs.leon_encoding(model_type, 
                                                    message, 
                                                    require_nodes=True, 
                                                    workload=workload, 
                                                    configs=configs, 
                                                    op_name_to_one_hot=op_name_to_one_hot,
                                                    plan_parameters=plan_parameters, 
                                                    feature_statistics=feature_statistics, 
                                                    sql=sqls_chunk[q_recieved_cnt])
                    elif model_type == "TreeConv":
                        encoded_plans, \
                        attns, \
                        queryfeature, \
                        nodes = envs.leon_encoding(model_type, 
                                                    message, 
                                                    require_nodes=True, 
                                                    workload=workload, 
                                                    queryFeaturizer=queryFeaturizer, 
                                                    nodeFeaturizer=nodeFeaturizer, 
                                                    sql=sqls_chunk[q_recieved_cnt])
    
                    if nodes is None:
                        continue
                
                    
                    # STEP 2) pick node to execute
                    ##############################execute node picked by leon#################
                    Subsitution(leon_node, nodes, Exp)
                    ##################################################################
                    
                    costs = torch.tensor(getNodesCost(nodes)).to(DEVICE)
                    model.model.to(DEVICE)
                    e_pad, a_pad = _batch(encoded_plans, attns)
                    cali_all = get_calibrations(model, queryfeature, e_pad, a_pad)
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                    ucb_idx = get_ucb_idx(cali_all, costs)
                    min_exec_num = min(len(ucb_idx), min_exec_num)
                    num_to_exe = max(min(math.ceil(pct * len(ucb_idx)), max_exec_num), min_exec_num)
                    
                    # Select the top num_to_exe nodes to execute by cost value
                    costs = costs.cpu().numpy()
                    sorted_indices = np.argsort(costs)
                    # Start with the minimum cost index
                    selected_indices = [sorted_indices[0]]
                    # Iterate over the sorted costs
                    for i in range(1, len(sorted_indices)):
                        # If the difference with the last selected cost is greater than 1.02, 
                        # add it to the selected indices
                        if (costs[sorted_indices[i]] / costs[selected_indices[-1]]) > 1.02:
                            selected_indices.append(sorted_indices[i])
                        if len(selected_indices) == num_to_exe:
                            break
                    if len(selected_indices) < num_to_exe:
                        for i in range(1, len(sorted_indices)):
                            if sorted_indices[i] not in selected_indices:
                                selected_indices.append(sorted_indices[i])
                                if len(selected_indices) == num_to_exe:
                                    break
                    costs_index = selected_indices

                    # STEP 3) execute with ENABLE_LEON=False and add exp
                    # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids], ...]
                    for i in range(num_to_exe):
                        node_idx = ucb_idx[i] # 第 i 大ucb 的 node idx
                        cost_index = costs_index[i]
                        a_node = nodes[node_idx]
                        b_node = nodes[cost_index]
                        # (1) add new EqSet key in exp
                        if i == 0: 
                            eqKey = a_node.info['join_tables']
                            Exp.AddEqSet(eqKey, curr_file[q_recieved_cnt])
                            # if Exp.GetEqSet()[eqKey].first_latency == TIME_OUT: # 不pick node后,直接删
                            #     Exp.DeleteOneEqset(eqKey)
                            #     break     
                        # (2) add experience of certain EqSet key
                        a_plan = a_node
                        b_plan = b_node
                        if not Exp.isCache(eqKey, a_plan) and \
                            not envs.CurrCache(exec_plan, a_plan):
                            PlanToExecute(
                                a_plan, eqKey, Exp, encoded_plans[node_idx], attns[node_idx])
        
                        if not Exp.isCache(eqKey, b_plan) and \
                            not envs.CurrCache(exec_plan, b_plan):
                            PlanToExecute(
                                b_plan, eqKey, Exp, encoded_plans[cost_index], attns[cost_index])
                    
                            
        print("Curr_Plan_Len: ", len(exec_plan))
        exec_plan = sorted(exec_plan, key=lambda x: x.cost, reverse=True)
        new_exec_plan = []
        for exec_one_plan in exec_plan:
            assert isinstance(exec_one_plan, ExecPlan)
            should_add = True
            cache_list = exp_cache.get(exec_one_plan.eq_set, None)
            if cache_list:
                for one_cache in cache_list:
                    cur_plan_node = exec_one_plan.plan
                    if one_cache.cost == cur_plan_node.cost and \
                    one_cache.info['sql_str'] == cur_plan_node.info['sql_str'] and \
                          one_cache.hint_str() == cur_plan_node.hint_str():
                        cur_plan_node.info['latency'] = one_cache.info['latency']
                        Exp.AppendExp(exec_one_plan.eq_set, exec_one_plan.plan)
                        should_add = False
                        break
            if should_add:
                new_exec_plan.append(exec_one_plan)
        # Plan Execution
        results = pool.map_unordered(actor_call_leon, new_exec_plan)
        loss_node = 0
        for result in results:
            if result == None:
                loss_node += 1
                continue
            Exp.AppendExp(result[1], result[0])
        if loss_node > 0:
            print("loss_node", loss_node)

        del queryfeature, encoded_plans, attns
        gc.collect()
        torch.cuda.empty_cache()

        ##########Delete EqSet#############
        Exp.DeleteEqSet()
        eqset = Exp.GetEqSet()
        print("len_eqset", Exp._getEqNum())
        logger.log_metrics({"len_eqset": Exp._getEqNum()}, step=my_step)
        for eq in eqset:
            print(f"{Exp.GetQueryId(eq)}Eq:{eq}," + \
                  f"len:{len(Exp.GetExp(eq))}," + \
                  f"opt_time:{round(Exp.GetEqSet()[eq].opt_time, 2)}," + \
                  f"eqset_latency:{round(Exp.GetEqSet()[eq].eqset_latency, 2)}")

        end_time = time.time()
        logger.log_metrics({"Time/pick_nodes_time": end_time - start_time}, step=my_step)
        start_time = time.time()

        # ++++ PHASE 3. ++++ model training
        train_pairs = Exp.Getpair()
        dnn_pairs = Exp.PreGetpair()

        logger.log_metrics({"train_pairs": len(train_pairs)}, step=my_step)
        print("len(train_pairs)" ,len(train_pairs))
        print("len(dnn_pairs)" ,len(dnn_pairs))
        if len(dnn_pairs) > min_batch_size:
            leon_dataset = prepare_dataset(
                dnn_pairs, True, nodeFeaturizer, Exp.GetEncoding())
            del dnn_pairs
            gc.collect()
            dataset_size = len(leon_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            train_ds, val_ds = torch.utils.data.random_split(leon_dataset, [train_size, val_size])
            dataloader_train = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=7)
            dataloader_val = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=7)
            dnn_model.optimizer_state_dict = dnn_prev_optimizer_state_dict
            torch.cuda.empty_cache()
            trainer = pl.Trainer(accelerator="gpu",
                                devices=[train_gpu],
                                enable_progress_bar=True,
                                max_epochs=100,
                                callbacks=[plc.EarlyStopping(
                                            monitor='val_acc',
                                            mode='max',
                                            patience=2,
                                            min_delta=0.001,
                                            check_on_train_epoch_end=False,
                                            verbose=True
                                        )],
                                logger=None)
            trainer.fit(dnn_model, dataloader_train, dataloader_val)
            del leon_dataset, train_ds, val_ds, dataloader_train, dataloader_val

        if len(train_pairs) > min_batch_size:
            leon_dataset = prepare_dataset(
                train_pairs, True, nodeFeaturizer, Exp.GetEncoding())
            del train_pairs
            gc.collect()
            dataset_size = len(leon_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            train_ds, val_ds = torch.utils.data.random_split(leon_dataset, [train_size, val_size])
            dataloader_train = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=7)
            dataloader_val = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=7)
            # dataset_test = BucketDataset(Exp.OnlyGetExp(), keys=Exp.GetExpKeys(), nodeFeaturizer=nodeFeaturizer, dict=encoding_dict)
            # batch_sampler = BucketBatchSampler(dataset_test.buckets, batch_size=1)
            # dataloader_test = DataLoader(dataset_test, batch_sampler=batch_sampler, num_workers=7)
            # model = load_model(model_path, prev_optimizer_state_dict).to(DEVICE)
            model.optimizer_state_dict = prev_optimizer_state_dict
            callbacks = load_callbacks(logger=None)
            torch.cuda.empty_cache()
            trainer = pl.Trainer(accelerator="gpu",
                                devices=[train_gpu],
                                enable_progress_bar=False,
                                max_epochs=100,
                                callbacks=callbacks,
                                logger=logger)
            trainer.fit(model, dataloader_train, dataloader_val)
            # trainer.test(model, dataloader_test)
            model.eq_summary = {key: 0 for key in eqset}
            prev_optimizer_state_dict = trainer.optimizers[0].state_dict()
            del leon_dataset, train_ds, val_ds, \
                dataloader_train, dataloader_val 
            #, dataset_test, batch_sampler, dataloader_test
            gc.collect()
            torch.cuda.empty_cache()

        print("*"*20)
        print("Current Accuracy For Each EqSet: ", model.eq_summary)
        print("*"*20)
        ch_start_idx = ch_start_idx + chunk_size
        # save model
        torch.save(dnn_model.model, "./log/dnn_model.pth")
        torch.save(model.model, model_path)
        ray.get(task_counter.reload_model.remote(eqset.keys(), model.eq_summary))
        # clear pkl
        if os.path.exists(message_path):
            os.remove(message_path)
            print(f"Successfully remove {message_path}")
        else:
            print(f"Fail to remove {message_path}")
        end_time = time.time()
        logger.log_metrics({"Time/train_time": end_time - start_time}, step=my_step)
        my_step += 1
        with open("./log/exp_v5.pkl", 'wb') as f:
            pickle.dump(Exp.OnlyGetExp(), f) 
        



    
