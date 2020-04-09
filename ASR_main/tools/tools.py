import pickle
import time

def save_pickle(obj, path = None):
    if path == None:
        path = time.strftime('%y%m%d_%H%M%S.p')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def sort_load(data_dir, load_func = None):
    '''sort and load data
    if load_func == None, defaults to pickle'''
    if load_func == None:
        load_func = load_pickle

    file_list = sorted(os.listdir(data_dir))
    loaded = list()

    for file in file_list:
        data_dir_1 = os.path.join(data_dir, file)
        data = load_func(data_dir_1)
        loaded.append(data)

    return loaded

def write(path, content, encoding = None):
    with open(path, 'w', encoding = encoding) as f:
        f.write(content)

def read(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.read()
    return text

def append(path, content, encoding = None):
    with open(path, 'a', encoding = encoding) as f:
        f.write(content)

def readlines(path, encoding = None):
    with open(path, 'r', encoding = encoding) as f:
        text = f.readlines()
    return text

def print_stars():
    print('*' * 50)

class Printer():
    def __init__(self, filewrite_dir = None):
        self.content = ''
        self.filewrite_dir = filewrite_dir

    def add(self, text):
        self.content += text

    def print(self, text, end='\n', flush=False):
        self.add(text)
        print(self.content, end=end, flush=flush)
        if self.filewrite_dir is not None:
            append(self.filewrite_dir, self.content + end)
        self.content=''

    def reset(self):
        self.content = ''

class Tree():

    def __init__(self, data = None, parent = None):
        self.data = data

        self.parent = parent
        self.children = list()
        self.depth = 0

    def __call__(self):
        return self.data

    def __getitem__(self, key):
        pass

    def __setitem__(self, key):
        pass
    def __len__(self):
        pass
