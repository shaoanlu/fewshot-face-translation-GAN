import yaml

def load_yaml(path_config):
    with open(path_config, 'r') as config:
        return yaml.safe_load(config)