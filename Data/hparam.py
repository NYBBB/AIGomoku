import yaml as ya


def load_hparam(path):
    stream = open(path, 'r', encoding='UTF-8')

    docs = ya.load_all(stream, Loader=ya.FullLoader)


    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
            pass
        pass
    return hparam_dict


# 创建一个字典类
class Dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        super().__init__()
        # 判断是否是空字典，如果是，则创建
        dct = dict() if not dct else dct

        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
                pass
            self[key] = value

    pass


class Hparam(Dotdict):

    def __init__(self, filepath='./Data/Data.yaml'):
        super(Dotdict, self).__init__()

        # 通过路径加载yaml文件做成字典
        # 当前得到的数据没法直接用“.”的方式去访问数据
        hp_dict = load_hparam(filepath)

        # 获得加强之后的字典，可以直接用“.”去访问参数
        hp_dotdict = Dotdict(hp_dict)

        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            pass

        pass

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


Data = Hparam()
