import os
from configurations.constant import Constant
from demos.utils import util

if __name__ == '__main__':
    project_path = Constant().get_project_path()

    filename = os.path.join(project_path, 'datasets', 'AishellSpeech', 'data_aishell.tgz')
    # 解压
    util.extract(filename)
