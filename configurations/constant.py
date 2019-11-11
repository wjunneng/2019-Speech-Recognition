import yaml
import os


class AttrDict(dict):
    """
    递归返回配置文件中的内容
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None

        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])

        return self[item]


class Constant(object):
    """
    读取配置文件中的内容
    """

    def __init__(self):
        self.project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.config_file_path = os.path.join(self.project_path, 'configurations', 'constant.yaml')

    def get_configuration(self):
        configuration = AttrDict(yaml.load(open(self.config_file_path)))

        return configuration

    def get_project_path(self):
        return self.project_path
