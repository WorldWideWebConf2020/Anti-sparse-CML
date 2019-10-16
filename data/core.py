import requests, zipfile, io, os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io
from tqdm import tqdm

def download_dataset(name):

    os.chdir('data')
    
    if not os.path.exists(name):
        os.mkdir(name)
    
    print('DOWNLOADING RAW FILE')
    
    if name == 'ml-1m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()

    elif name == 'amazon-CDs':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv'
        df = pd.read_csv(url, header=['user-id', 'item-id', 'rating', 'timestamp'])
        df.to_csv(name)

    else:
        print('Download method for dataset {} not supported'.format(name))
    
    os.chdir('..')

def process_data(name, train_size=0.5, eval_size=0.25,
              min_pos_interactions=16, min_rating=4, seed=42):

    rd_state = np.random.RandomState(seed=seed)

    print('READING RAW FILE')

    if name == 'ml-1m':
        df = pd.read_csv('data/ml-1m/ratings.dat', delimiter='::', header=None).iloc[:, :3]
        df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)
    elif name == 'amazon-CDs':
        df = pd.read_csv('data/amazon-CDs/ratings.csv', header=None).iloc[:, :3]
        df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)
    elif name == 'yelp':
        df = pd.read_csv('data/yelp/review.csv', usecols=[1, 2, 3])
        df.rename(columns={'business_id': 'item_id', 'stars': 'rating'}, inplace=True)
    else:
        raise NotImplementedError('{} dataset processing was not implemented yet'.format(name))

    df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)
    ratings = df['rating'].values

    _, user_ids = np.unique(df['user_id'], return_inverse=True)
    _, item_ids = np.unique(df['item_id'], return_inverse=True)
    m, n = np.max(user_ids)+1, np.max(item_ids)+1

    rating_matrix = csr_matrix((ratings, (user_ids, item_ids)), shape=(m, n))
    pos_inters_matrix = (rating_matrix >= min_rating)
    pos_inters_matrix.eliminate_zeros()
    n_pos_inters_per_u = pos_inters_matrix.getnnz(axis=1)
    mask_users = (n_pos_inters_per_u >= min_pos_interactions)
    rating_matrix = rating_matrix[mask_users, :]
    m, n = rating_matrix.shape

    print('Num users: {}'.format(m))
    print('Num items: {}'.format(n))
    print('Num interactions: {}'.format(rating_matrix.getnnz()))

    train_triplets = {'n_items': [0], 'item_ids':[], 'ratings': []}
    eval_triplets = {'n_items': [0], 'item_ids':[], 'ratings': []}
    test_triplets = {'n_items': [0], 'item_ids':[], 'ratings': []}

    for inters in tqdm(rating_matrix,
                       desc='SPLITTING TRAIN-EVAL-TEST'):
        ratings = inters.data
        i_ids = inters.indices
        n_inters = len(ratings)
        indices = np.arange(n_inters)
        rd_state.shuffle(indices)
        n_train = int(train_size*n_inters)
        n_eval = int(eval_size*n_inters)
        train_indices, eval_indices, test_indices = np.array_split(indices,
                                                    [n_train, n_eval + n_train])
        train_item_ids = i_ids[train_indices]
        train_ratings = ratings[train_indices]
        eval_item_ids = i_ids[eval_indices]
        eval_ratings = ratings[eval_indices]
        test_item_ids = i_ids[test_indices]
        test_ratings = ratings[test_indices]

        train_triplets['n_items'].append(len(train_item_ids))
        train_triplets['item_ids'] += train_item_ids.tolist()
        train_triplets['ratings'] += train_ratings.tolist()

        eval_triplets['n_items'].append(len(eval_item_ids))
        eval_triplets['item_ids'] += eval_item_ids.tolist()
        eval_triplets['ratings'] += eval_ratings.tolist()

        test_triplets['n_items'].append(len(test_item_ids))
        test_triplets['item_ids'] += test_item_ids.tolist()
        test_triplets['ratings'] += test_ratings.tolist()

    train_ratings = csr_matrix((train_triplets['ratings'],
                                train_triplets['item_ids'],
                                np.cumsum(train_triplets['n_items'])),
                                shape=(m,n))
    eval_ratings = csr_matrix((eval_triplets['ratings'],
                               eval_triplets['item_ids'],
                               np.cumsum(eval_triplets['n_items'])),
                               shape=(m,n))
    test_ratings = csr_matrix((test_triplets['ratings'],
                               test_triplets['item_ids'],
                               np.cumsum(test_triplets['n_items'])),
                               shape=(m,n))

    print('SAVING SPARSE MATRICES TO SPARSE .MAT')

    scipy.io.savemat('data/' + name + '/ratings.mat',
                     {'train_ratings': train_ratings.data,
                     'eval_ratings': eval_ratings.data,
                     'test_ratings': test_ratings.data,
                     'user_ids_train': train_ratings.nonzero()[0],
                     'user_ids_eval': eval_ratings.nonzero()[0],
                     'user_ids_test': test_ratings.nonzero()[0],
                     'item_ids_train': train_ratings.nonzero()[1],
                     'item_ids_eval': eval_ratings.nonzero()[1],
                     'item_ids_test': test_ratings.nonzero()[1]})

    return train_ratings, eval_ratings, test_ratings

def load_mat(matfile_path):
    res = scipy.io.loadmat(matfile_path)
    train_ratings = res['train_ratings'].reshape(-1)
    eval_ratings = res['eval_ratings'].reshape(-1)
    test_ratings = res['test_ratings'].reshape(-1)
    user_ids_train = res['user_ids_train'].reshape(-1)
    user_ids_eval = res['user_ids_eval'].reshape(-1)
    user_ids_test = res['user_ids_test'].reshape(-1)
    item_ids_train = res['item_ids_train'].reshape(-1)
    item_ids_eval = res['item_ids_eval'].reshape(-1)
    item_ids_test = res['item_ids_test'].reshape(-1)
    train_ratings_csr = csr_matrix((train_ratings, (user_ids_train, item_ids_train)))
    eval_ratings_csr = csr_matrix((eval_ratings, (user_ids_eval, item_ids_eval)))
    test_ratings_csr = csr_matrix((test_ratings, (user_ids_test, item_ids_test)))
    return train_ratings_csr, eval_ratings_csr, test_ratings_csr

if __name__ == '__main__':
    dataset_name = 'ml-1m'
    download_dataset(dataset_name)
    process_data(dataset_name)
