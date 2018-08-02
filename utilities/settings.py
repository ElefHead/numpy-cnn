layer_nums = {
    'fc': 1,
    'conv': 1
}
network_name = None
models_path = 'models'


def init():
    global layer_nums
    layer_nums = {
        'fc': 1,
        'conv': 1
    }
    global network_name
    network_name = None


def get_layer_num(layer_type):
    global layer_nums
    return layer_nums[layer_type]


def get_models_path():
    return models_path


def inc_layer_num(layer_type):
    global layer_nums
    layer_nums[layer_type] += 1


def set_network_name(name):
    global network_name
    network_name = name


def get_network_name():
    if network_name is None:
        raise RuntimeError("Model name not set, set name as 'model.set_name(<name>)'")
    return network_name
