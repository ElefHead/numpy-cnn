def init():
    global layer_nums
    layer_nums = {
        'fc': 0,
        'conv': 0
    }
    global network_name
    network_name = None


def get_layer_num(layer_type):
    global layer_nums
    return layer_nums[layer_type]


def inc_layer_num(layer_type):
    global layer_nums
    layer_nums[layer_type] += 1


def set_network_name(name):
    global network_name
    network_name = name


def get_network_name():
    return network_name
