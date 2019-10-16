from eval import eval_routine
from models.Pairwise.cml import AntiSparseCML
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
max_users_eval = 7000
datasets = ['ml-1m']
r = ['r=8', 'r=16', 'r=32', 'r=64', 'r=128', 'r=256']

eval_routine(AntiSparseCML, 'anti_sparse_cml', datasets, r, max_users_eval,
            gpu_options)
