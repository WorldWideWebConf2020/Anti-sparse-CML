import numpy as np
import os
from scipy.sparse import csr_matrix
import tensorflow as tf
from data.core import load_mat
import re
import json

def ndcg_at_k(pos_inters_csr, seen_inters_csr, pdists, k, verbose=False):

    # Compute discounts
    discounts = np.log(np.arange(2, 2+k))

    # Discard seen interactions
    csr_masked = pos_inters_csr.multiply(np.logical_not((seen_inters_csr>0).toarray()))

    # Mask of users with no remaining positive interactions
    n_pos_inter_per_u = np.count_nonzero(csr_masked.toarray(), axis=1)
    mask_non_isolated_users = (n_pos_inter_per_u > 0)

    # Print total number of discarded users
    if verbose:
        n_discarded = np.sum(np.logical_not(mask_non_isolated_users))
        print('{} users were discarded to compute NDCG@{}'.format(n_discarded, k))

    # Filter
    f_pos = pos_inters_csr[mask_non_isolated_users, :]
    f_train = seen_inters_csr[mask_non_isolated_users, :]
    f_pdists = pdists[mask_non_isolated_users, :]
    f_n_pos_inter_per_u = n_pos_inter_per_u[mask_non_isolated_users]

    # Fill taboo pdists with high values
    f_pdists[np.where((f_train>0).toarray())] = 1e20

    # Get top k pdists
    # Partials
    partial_argsort = np.argpartition(f_pdists, k, axis=1)[:, :k]
    partial_sort = np.take_along_axis(f_pdists, partial_argsort, axis=1)

    # Total
    k_sorted_pdists = np.sort(partial_sort, axis=1)
    k_sorted_items = np.take_along_axis(partial_argsort,
                                        np.argsort(partial_sort, axis=1), axis=1)

    # Take relevance for remaining users
    relevance_k = np.take_along_axis(f_pos.toarray(),
                                     k_sorted_items, axis=1).astype(float)

    # Compute dcg
    dcg = np.sum(relevance_k / discounts[None, :], axis=1)

    # Create idcg table
    table_idcg = np.cumsum(1/discounts)

    # Compute idcg
    idcg = table_idcg[np.minimum(f_n_pos_inter_per_u, k) - 1]

    return (dcg/idcg).tolist()

def gen_pos_taboos(a, b, c, min_rating=4):
    seen_inters_train = a.nonzero()
    seen_inters_test = b.nonzero()
    all_seens = np.hstack([seen_inters_train, seen_inters_test])
    taboo_csr = csr_matrix((np.ones(len(all_seens[0])),
                            (all_seens[0], all_seens[1])),
                            shape=c.shape)
    pos_csr = (c>=min_rating)

    return pos_csr, taboo_csr

def restore_model(model_class, sess, model_dir):
    hyper_path = model_dir + '/hyperparams.json'
    f = open(hyper_path, 'r')
    hyper_dict = json.load(f)
    f.close()
    model = model_class(hyper_dict)
    model._instantiate_model(sess, None)
    return model

def restore_best_model(model_class, model_name, sess, dataset, dim):
    models_dir = os.path.join('results', dataset, model_name, dim)
    best_ndcg = 0
    for dir in os.listdir(models_dir):
        print('Hyperparams: {}'.format(dir))
        try:
            f = open(os.path.join(models_dir, dir, 'metrics.txt'), 'r')
            ndcg = float(re.findall("\d+\.\d+", f.read())[0])
            print(ndcg)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                dir_best_model = dir
        except Exception as e:
            print(e)
    return restore_model(model_class, sess, os.path.join(models_dir, dir_best_model))

def eval_routine(model_class, model_name, datasets,
                dims, max_users_eval,gpu_options):

    for dataset in datasets:
        print('Evaluated model: {}'.format(model_name))
        print('----------------------------------------------')
        print('Retrieving socres from: ' + dataset)
        ratings_path = os.path.join('data', dataset)
        tr_csr, eval_csr, test_csr = load_mat(os.path.join(ratings_path, 'ratings.mat'))
        pos, taboo = gen_pos_taboos(tr_csr, eval_csr, test_csr)
        n_users = tr_csr.shape[0]
        n_chunks = n_users//max_users_eval + 1
        chunk_list = np.array_split(np.arange(n_users), n_chunks)
        for dim in dims:
            print('----------------------------------------------')
            print(dim)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                best_model = restore_best_model(model_class, model_name,
                                                sess, dataset, dim)
                ndcg = []
                for chunk in chunk_list:
                    pdists = sess.run(best_model.pdists(chunk))
                    ndcg += ndcg_at_k(pos[chunk, :], taboo[chunk, :], pdists, 10)
                ndcg_test = np.mean(ndcg)
            tf.reset_default_graph()
            line = 'NDCG@10 on test set: {}'.format(ndcg_test)
            print(line)
            f = open(os.path.join('results', dataset, model_name,
                                  dim, 'metrics.txt'), 'w')
            f.write(line)
            f.close()
