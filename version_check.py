import tensorflow as tf
print(tf.__version__)

import  tensorflow.python.platform.build_info as build
print(build.build_info)
print(build.build_info['cuda_version'])
print(build.build_info['cudnn_version'])
