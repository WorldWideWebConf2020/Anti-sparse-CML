from models.Pairwise.cml import SignCML
import numpy as np
from data.core import load_mat
from eval import ndcg_at_k
import os
import tensorflow as tf
from models import restore_model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)

datasets = ['ml-1m']

r_list = [8, 16, 32, 64, 128, 256]
margins = [0.25, 0.5, 0.75, 1]

params = {}
params['seed'] = 42
params['n_epochs'] = 100
params['n_negatives'] = 1
params['batch_size'] = 512
params['lr'] = 7.5e-4
params['n_users_eval'] = 7000
params['eval_every'] = 10
data_dir =  'data'
saving_dir = 'results'

for dataset in datasets:
    mat_file = os.path.join(data_dir, dataset, 'ratings')
    train_ratings, eval_ratings, test_ratings = load_mat(mat_file)
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
        for margin in margins:
            print('MARGIN = {}'.format(margin))
            params['margin'] = margin
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sign_cml = SignCML(params)
                sign_cml.fit(sess, train_ratings, eval_ratings, test_ratings, save=True)
            tf.reset_default_graph()
