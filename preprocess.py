# Author claude
# Data 17-08-24
import numpy as np


# Just for label string
class IndexedSet:
    def __init__(self):
        self.data = dict()
        self.max_index = -1

    def add(self, item):
        if self.data.get(item) is None:
            self.max_index += 1
            self.data[item] = self.max_index

    def label(self, item):
        result = self.data.get(item)
        if result is None:
            return -1
        return result


protocol = IndexedSet()
service = IndexedSet()
flag = IndexedSet()
error = IndexedSet()


def __protocol_converter(value):
    protocol.add(value)
    return protocol.label(value)


def __service_converter(value):
    service.add(value)
    return service.label(value)


def __flag_converter(value):
    flag.add(value)
    return flag.label(value)


def __error_converter(value):
    error.add(value)
    return error.label(value)


converter_dict = {
        1: __protocol_converter, 
        2: __service_converter, 
        3: __flag_converter, 
        -1: __error_converter, 
}


# load data from file
def load_data(filename, converter=converter_dict):
    X = np.loadtxt(filename, delimiter=',', converters=converter)
    return X


def __test():
    load_data('kddcup.data_10_percent')
    pass


if __name__ == '__main__':
    __test()
