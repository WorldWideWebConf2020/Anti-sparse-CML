from models.Pointwise.ch import CH
import numpy as np
from data.core import load_mat
from eval import ndcg_at_k
import os
import tensorflow as tf
from models import restore_model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
r_list = [8, 16, 32]
lambd_list = [1e3, 1e2, 1e1, 1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
datasets = ['ml-1m']

params = {}
params['seed'] = 42
params['n_epochs'] = 20
params['n_users_eval'] = 7000
params['eval_every'] = 1

data_dir =  'data/'
saving_dir = 'results/'

for dataset in datasets:
    train_ratings, eval_ratings, test_ratings = load_mat(data_dir + dataset + '/ratings.mat')
    m, n = train_ratings.shape
    n_inters = len(train_ratings.data)
    print('Number of users/items: {}/{}'.format(m, n))
    print('Number of positive train interactions: {}'.format(n_inters))
    params['saving_dir'] = os.path.join(saving_dir, dataset)
    for r in r_list:
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('EMBEDDING DIM = {}'.format(r))
        params['r'] = r
        for lambd in lambd_list:
            params['lambda'] = lambd
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                ch = CH(params)
                ch.fit(sess, train_ratings, eval_ratings, test_ratings, save=True)
            tf.reset_default_graph()
