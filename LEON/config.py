import configparser

def read_config(section='leon', path='./conf/leon.cfg'):
    config = configparser.ConfigParser()
    config.read(path)

    # if section not in config:
    #     print(f"leon.cfg does not have a [{section}] section.")
    #     exit(-1)

    # config = config[section]
    return config
