import json
import socketserver
import util.envs as envs
import util.plans_lib as plans_lib
import torch
import os
import re
import time
from models.Transformer import *
from models.DNN import *
import models.Treeconv as treeconv
from leon_experience import TIME_OUT
import json
import pickle
import ray
import uuid
from config import read_config
import torch.nn.functional as F
conf = read_config()

DEVICE = 'cpu'

def is_subset(s1, s2):
    s1 = set(s1.split(','))
    s2 = set(s2.split(','))
    if abs(len(s1) - len(s2)) != 1:
        return False
    result1 = s1.issubset(s2)
    result2 = s2.issubset(s1)
    return result1 or result2

@ray.remote
class Communicator:
    """
    Communicator is a class that is used to communicate with the leon server and leon trainer.
    """
    def __init__(self):
        self.recieved_task = 0
        self.eq_summary = dict()
        train_file, training_query = envs.load_train_files(conf['leon']['workload_type'])
        self.eqset = envs.find_alias(training_query)
        self.online_flag = False # True表示要message False表示不要
        port_list = eval(conf['leon']["other_leon_port"]) + [int(conf['leon']["Port"])]
        print("Use Port List: ", port_list)
        self.port_dict = {key: True for key in port_list}

    def Add_task(self):
        self.recieved_task += 1
    
    def GetRecievedTask(self):
        return self.recieved_task
    
    def load_model(self, port):
        temp = self.port_dict[port]
        self.port_dict[port] = False
        return temp , self.eqset, self.eq_summary
    
    def reload_model(self, eqset, eq_summary):
        self.port_dict = {key: True for key in self.port_dict}
        self.eqset = eqset
        self.eq_summary = eq_summary

    def GetOnline(self):
        return self.online_flag
    
    def WriteOnline(self, flag: bool):
        self.online_flag = flag
    
@ray.remote
class FileWriter:
    """
    FileWriter is a class that is used to write messages to a file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed_tasks = 0
        self.recieved_task = 0

    def write_file(self, nodes):
        try:
            with open(self.file_path, 'ab') as file:
                pickle.dump(nodes, file)
                print("write one message")
            self.completed_tasks += 1
        except Exception as e:
            print("write_file() fail to open file and to write message", e)

    def Add_task(self):
        self.recieved_task += 1

    def GetCompletedTasks(self):
        return self.completed_tasks
    
    def complete_all_tasks(self):
        if self.completed_tasks == self.recieved_task:
            return True
        else:
            return False
    
def _batch(trees, indexes, padding_size=200):
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

class LeonModel:

    def __init__(self, model_port):
        self.__model = None
        GenerateUniqueNameSpace = lambda: str(uuid.uuid4())
        namespace = GenerateUniqueNameSpace()
        # unique namespace for ray cluster
        with open('./conf/namespace.txt', 'w') as f:
            f.write(namespace)
        # ray should be init in sub process
        context = ray.init(namespace=namespace, 
                           _temp_dir= conf['leon']['ray_path'] + "/log/ray")   
        print(context.address_info)
        ray_address = context.address_info['address']
        with open('./conf/ray_address.txt', 'w') as f:
            f.write(ray_address)
        node_path = "./log/messages.pkl"
        # Initialize the ray actor: Communicator and FileWriter
        self.writer_hander = FileWriter.options(name="leon_server").remote(node_path)
        self.communicator = Communicator.options(name="counter").remote()
        self.eqset = None

        # ML model configuration
        self.channels = channels
        self.workload = envs.wordload_init(conf['leon']['workload_type'])
        self.queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)

        self.model_type = conf['leon']['model_type']
        if self.model_type == "TreeConv":
            self.nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(self.workload.workload_info)
        elif self.model_type == "Transformer":
            statistics_file_path = "./statistics.json"
            self.feature_statistics = load_json(statistics_file_path)
            add_numerical_scalers(self.feature_statistics)
            self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)
        
        self.plan_channel_init = plan_channel_init(self.workload)

        # Query Optimization Configuration
        self.eq_summary = dict() # Training Result
        self.current_eq_summary = None # Current Query's Eq Summary
        self.Current_Level = None # Current planning level
        self.Levels_Needed = None # How many levels are needed for the query
        self.Query_Id = None  # Current Query's ID/Name
        self.Old_Query_Id = None # Last Query's ID/Name
        # Whether to explain the query
        # we explain the query when the query is new (before the actual execution)
        # to make sure the query has consecutive levels of equivalent sets
        self.explain_flag = False 
        self.Old_Current_Level = None 
        self.continuous_eqset = []
        self.model_port = model_port
        print("Init LeonModel Finish")
            
    def get_calibrations(self, seqs, attns, QueryFeature):
        with torch.no_grad():
            self.__model.eval()
            QueryFeature = QueryFeature.to(DEVICE)
            seqs = seqs.to(DEVICE)
            attns = attns.to(DEVICE)
            if self.model_type == "TreeConv":
                cali_all = torch.tanh(self.__model(QueryFeature, seqs, attns)).add(1).squeeze(1)
            elif self.model_type == "Transformer":
                cali = self.__model(seqs, attns, QueryFeature) # cali.shape [# of plan, pad_length]
                cali_all = cali[:, 0] # [# of plan] -> [# of plan, 1] cali_all plan
        
        return cali_all
    
    def encoding(self, X): 
        if self.model_type == "Transformer":
            encoded_plans, attns, queryfeature, _ = envs.leon_encoding(self.model_type, X, 
                                                                           require_nodes=False, workload=self.workload, 
                                                                           configs=configs, op_name_to_one_hot=self.op_name_to_one_hot,
                                                                           plan_parameters=plan_parameters, feature_statistics=self.feature_statistics)
            return encoded_plans, attns, queryfeature

        elif self.model_type == "TreeConv":
            trees, indexes, queryfeature, nodes = envs.leon_encoding(self.model_type, 
                                                                     X, 
                                                                     require_nodes=True, 
                                                                     workload=self.workload, 
                                                                     queryFeaturizer=self.queryFeaturizer, 
                                                                     nodeFeaturizer=self.nodeFeaturizer)
            if isinstance(trees, list):
                trees, indexes, = _batch(trees, indexes)
            featurizeDNN(nodes, self.plan_channel_init)
            return trees, indexes, queryfeature, nodes
        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")
    
    def inference(self, seqs, attns, QueryFeature=None, nodes=None):
        cali_all = self.get_calibrations(seqs, attns, QueryFeature)

        def diff_normalized(p1, p2):
            norm = torch.sum(p1, dim=1)
            pair = (p1 - p2) / norm.unsqueeze(1)
            return pair
        
        def format_scientific_notation(number):
            """
            Format a number to scientific notation.
            For example, 9995 -> 9.995,1,4
            """
            str_number = "{:e}".format(number)
            mantissa, exponent = str_number.split('e')
            mantissa = 9.994 if float(mantissa) >= 9.995 else float(mantissa)
            mantissa = format(mantissa, '.2f')
            exponent = int(exponent)
            exponent = max(-9, min(9, exponent))
            result = "{},{},{:d}".format(mantissa, '1' if exponent >= 0 else '0', abs(exponent))
            return result
        
        cost_node = nodes[0]
        for i, node in enumerate(nodes):
            self.__dnn_model.eval()
            if self.__dnn_model(diff_normalized(node.info['dnn_vec'], cost_node.info['dnn_vec']).to(torch.float32)).squeeze(0)[0] > 0.65:
                if i != 0:
                    cali_all[i] = TIME_OUT # 1000000
      
        cali_str = [format_scientific_notation(i) for i in cali_all.tolist()] # 最后一次 cali
        cali_strs = ';'.join(cali_str)
        return cali_strs
    
    def load_model(self, path):
        if not os.path.exists(path):
            if self.model_type == "Transformer":
                print("load transformer model")
                model = SeqFormer(
                    input_dim=configs['node_length'],
                    hidden_dim=256,
                    output_dim=1,
                    mlp_activation="ReLU",
                    transformer_activation="gelu",
                    mlp_dropout=0.1,
                    transformer_dropout=0.1,
                    query_dim=configs['query_dim'],
                    padding_size=configs['pad_length']
                    ).to(DEVICE) # 
            elif self.model_type == "TreeConv":
                print("load treeconv model")
                model = treeconv.TreeConvolution(666, 50, 1).to(DEVICE)
            torch.save(model, path)
        else:
            model = torch.load(path, map_location=DEVICE)
            dnn_model = torch.load("./log/dnn_model.pth", map_location=DEVICE)
            print(f"load checkpoint {path} Successfully!")
        return model, dnn_model
    
    def infer_equ(self, messages):
        reload_model, self.eqset, self.eq_summary = ray.get(self.communicator.load_model.remote(self.model_port))
        if reload_model:
            print(self.eqset)
            self.__model, self.__dnn_model = self.load_model("./log/model.pth")
        X = messages
        if not isinstance(X, list):
            X = [X]
        Relation_IDs = X[0]['Relation IDs']
        self.Current_Level = X[0]['Current Level']
        self.Levels_Needed = X[0]['Levels Needed']
        self.Query_Id = X[0]['QueryId']
        # Runing a new query
        if self.Query_Id != self.Old_Query_Id:
            self.Old_Query_Id = self.Query_Id
            self.continuous_eqset = []
            self.explain_flag = True

        out = ','.join(sorted(Relation_IDs.split()))
        
        if (out in self.eqset) and \
        (self.explain_flag or (out in self.continuous_eqset)): # and (self.Current_Level == self.Levels_Needed)
            # Find the continuous eqset
            if self.explain_flag:
                if self.continuous_eqset == []:
                    self.continuous_eqset.append(out)
                    self.Old_Current_Level = self.Current_Level
                else:
                    if self.Current_Level - self.Old_Current_Level <= 1:
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                    else:
                        self.continuous_eqset = []
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                # If the current level is the last level, 
                # we stop appending the continuous eqset
                # Then we find eqsets that are subset of others
                if self.Current_Level == self.Levels_Needed:
                    self.explain_flag = False
                    temp_list = [0] * len(self.continuous_eqset)
                    temp_list[-1] = 1
                    for i, eq_1 in enumerate(self.continuous_eqset):
                        for eq_2 in self.continuous_eqset[i+1:]:
                            if is_subset(eq_1, eq_2):
                                temp_list[i] = 1
                    new_list = []
                    for i in range(0, len(self.continuous_eqset)):
                        if temp_list[i]:
                            new_list.append(self.continuous_eqset[i])
                    self.continuous_eqset = new_list
                    print(self.continuous_eqset)
            if not self.eq_summary:
                self.current_eq_summary = 1
            else:
                self.current_eq_summary = self.eq_summary.get(out)
            self.curr_eqset = out
            print(X[0]['QueryId'], out)
            print(self.Current_Level, self.Levels_Needed)
            return '1'
        else:
            return '0'


    def predict_plan(self, messages):
        # json解析
        print("Predicting plan for ", len(messages), f" On Worker {self.model_port}")
        X = messages
        if not isinstance(X, list):
            X = [X]
    
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        # This is offline mode, we only need to execute the picked plan
        # picknode:{Relid};{Cost};
        if self.Query_Id.startswith("picknode:"):
            temp_id = self.Query_Id[len("picknode:"):]
            parts = temp_id.split(";")
            curr_level = parts[0]
            pick_plan = float(parts[1])
            if curr_level == self.curr_eqset:
                # If we pick the node (cost is equal to the pick_plan),
                # we give it a the lowest cost
                return ';'.join(['1.00,1,0' if i['Plan']['Total Cost'] != pick_plan \
                                  else '0.01,0,9' for i in X]) + ';'
        try:
            # Write the message to a file, at the offline mode
            # We only need to write the message to a file
            if ray.get(self.communicator.GetOnline.remote()) and self.Current_Level == self.Levels_Needed:
                self.communicator.Add_task.remote()
                self.writer_hander.write_file.remote(X)
        except:
            print("The ray writer_hander cannot write file.")

        # Validation Accuracy    
        if self.current_eq_summary is None:
            return ';'.join(['1.00,1,0' for _ in X]) + ';'

        # Encoding
        seqs, attns, QueryFeature, nodes = self.encoding(X)

        # Inference
        cali_strs = self.inference(seqs, attns, QueryFeature, nodes)
        return cali_strs + ';'
    

class SimpleLeonModel(LeonModel):
    """Minimal Leon Model for inference only."""

    def __init__(self, model_port):
        # 初始化
        self.__model = None
        with open ("./conf/namespace.txt", "r") as file:
            namespace = file.read().replace('\n', '')
        with open ("./conf/ray_address.txt", "r") as file:
            ray_address = file.read().replace('\n', '')
        context = ray.init(address=ray_address, namespace=namespace, _temp_dir=conf['leon']['ray_path'] + "/log/ray") # init only once
        print(context.address_info)
        
        self.eqset = None
        self.channels = channels
        self.workload = envs.wordload_init(conf['leon']['workload_type'])
        self.queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)
        self.model_type = conf['leon']['model_type']
        if self.model_type == "TreeConv":
            self.nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(self.workload.workload_info)
        elif self.model_type == "Transformer":
            statistics_file_path = "./statistics.json"
            self.feature_statistics = load_json(statistics_file_path)
            add_numerical_scalers(self.feature_statistics)
            self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)

        self.plan_channel_init = plan_channel_init(self.workload)

        self.communicator = ray.get_actor('counter')
        self.eq_summary = dict()
        self.current_eq_summary = None
        self.Current_Level = None
        self.Levels_Needed = None
        self.Query_Id = None
        self.Old_Query_Id = None
        self.explain_flag = False
        self.Old_Current_Level = None
        self.continuous_eqset = []
        self.model_port = model_port
        print("Init SimpleLeonModel Finish ")


class JSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            # Read data from the client
            str_buf += self.request.recv(81960).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            
            if (null_loc := str_buf.find("\n")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:    
                        def fix_json_msg(json):
                            pattern = r'(ANY|ALL) \((.*?):text\[\]\)'
                            matches = re.findall(pattern, json)
                            for _, match in matches:
                                extracted_string = match
                                cleaned_string = extracted_string.replace('"', '')
                                json = json.replace(extracted_string, cleaned_string)
                            return json
                        json_msg = fix_json_msg(json_msg)
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON:", repr(json_msg))
                        self.handle_json([])
                        break

class LeonJSONHandler(JSONTCPHandler):
    def setup(self):
        self.__messages = []
    
    def handle_json(self, data):
        if "final" in data:
            message_type = self.__messages[0]["type"]
            self.__messages = self.__messages[1:]
            if message_type == "query":
                result = self.server.leon_model.predict_plan(self.__messages)
                response = str(result).encode()
                try:
                    self.request.sendall(response)
                except Exception as e:
                    print(f"发送响应时出错：{e}")
                finally:
                    self.request.close()
            elif message_type == "should_opt":
                result = self.server.leon_model.infer_equ(self.__messages)
                response = str(result).encode()
                self.request.sendall(response)
                self.request.close()
            else:
                print("Unknown message type:", message_type)
            return True
        
        self.__messages.append(data)
        return False

def start_server(listen_on, port):
    model = LeonModel(port)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), LeonJSONHandler) as server:
        server.leon_model = model
        server.serve_forever()

def other_server(listen_on, port):
    model = SimpleLeonModel(port)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), LeonJSONHandler) as server:
        server.leon_model = model
        server.serve_forever()


if __name__ == "__main__":
    from multiprocessing import Process
    from config import read_config
    torch.multiprocessing.set_start_method('spawn')


    config = read_config()
    port = int(config['leon']["Port"])
    listen_on = config['leon']["ListenOn"]
    other_leon_port = eval(config['leon']["other_leon_port"])
    print(f"Listening on {listen_on} port {port}")
    
    server = Process(target=start_server, args=[listen_on, port])
    othersevers = []
    for i in other_leon_port:
        othersevers.append(Process(target=other_server, args=[listen_on, i]))

    print("Spawning server process...")
    server.start()
    time.sleep(10)
    for otherserver in othersevers:
        otherserver.start()
