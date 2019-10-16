from eval import eval_routine
from models.Pointwise.ch import CH
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
max_users_eval = 7000
datasets = ['ml-1m']
r = ['r=8']

eval_routine(CH, 'ch', datasets, r, max_users_eval, gpu_options)
