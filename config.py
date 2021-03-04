import yaml


class Config(dict):
    def __init__(self, filename):
        super().__init__()
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        for k, v in cfg_dict.items():
            self.__dict__[k] = v
