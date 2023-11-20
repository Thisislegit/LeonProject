import configparser

def read_config():
    config = configparser.ConfigParser()
    config.read("leon.cfg")

    if "leon" not in config:
        print("bao.cfg does not have a [leon] section.")
        exit(-1)

    config = config["leon"]
    return config
