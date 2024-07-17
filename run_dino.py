import sys

import gin
from procedures.dino_training import DINOTrainingProcedure

"""
To run training execute this script with gin config file as an argument.
"""


if __name__ == '__main__':
    config_file = sys.argv[1]
    gin.parse_config_file(config_file)
    tp = DINOTrainingProcedure()
    tp.run()
