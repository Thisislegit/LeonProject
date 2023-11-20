import json
import struct
import socketserver
from utils import *
import re

class LeonModel:

    def __init__(self):
        self.__model = None
    
    def load_model(self, path):
        pass
    
    def predict_plan(self, messages):
        print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        print(X[0])
        # seqs = [get_plan_seq_adj(x['Plan']) for x in X]
        # print(seqs[0])
        # print(op_names)
        # seqs_encoding = [generate_seqs_encoding(x) for x in seqs]
        # print(seqs_encoding[0])
        return ';'.join(['1.00,1,0' for _ in X]) + ';'

class JSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            # 这里只有断连才会退出
            str_buf += self.request.recv(1024).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            
            if (null_loc := str_buf.find("\n")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:    
                        def fix_json_msg(json):
                            pattern = r'ANY \((.*?):text\[\]\)'
                            matches = re.findall(pattern, json)
                            for match in matches:
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
                # self.request.sendall(struct.pack("I", result))
                self.request.sendall(response)
                self.request.close()
            elif message_type == "should_opt":
                print(self.__messages)
                response = str("1").encode()
                self.request.sendall(response)
                self.request.close()
            else:
                print("Unknown message type:", message_type)
            return True
        
        self.__messages.append(data)
        return False

def start_server(listen_on, port):
    model = LeonModel()

    # if os.path.exists(DEFAULT_MODEL_PATH):
    #     print("Loading existing model")
    #     model.load_model(DEFAULT_MODEL_PATH)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), LeonJSONHandler) as server:
        server.leon_model = model
        server.serve_forever()


if __name__ == "__main__":
    from multiprocessing import Process
    from config import read_config

    config = read_config()
    port = int(config["Port"])
    listen_on = config["ListenOn"]

    print(f"Listening on {listen_on} port {port}")
    
    server = Process(target=start_server, args=[listen_on, port])
    
    print("Spawning server process...")
    server.start()