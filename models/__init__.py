from tqdm import tqdm
import numpy as np
import tensorflow as tf
import json
from eval import ndcg_at_k, gen_pos_taboos, restore_model
import os

LONG_STRING = '--------------------------------------------------------------\n'

class Model(object):

    def __init__(self, params):
        self.params = params
        # Get basic model config
        self.n_epochs = params['n_epochs']
        self.saving_dir = params['saving_dir']
        self.r = params['r']
        self.seed = params['seed']
        self.n_users_eval = params['n_users_eval']
        self.eval_every = params['eval_every']
        # Set both np and tf random states
        self.state = np.random.RandomState(self.seed)
        tf.random.set_random_seed(self.seed)
        self.is_instantiated = False

    def fit(self, sess, train_ratings, eval_ratings,
                        test_ratings, save=False):

        # Save model hyperparams
        if save:
            hyper_save_path = self.model_dir + '/hyperparams.json'
            print('SAVING HYPERPARAMS in FILE {}'.format(hyper_save_path))
            hyperparams = json.dumps(self.params)
            f = open(hyper_save_path, 'w')
            f.write(hyperparams)
            f.close()

        # Solely evaluate on eval interactions, mask train and test
        pos_eval_csr, taboo_csr = gen_pos_taboos(train_ratings,
                                                test_ratings,
                                                eval_ratings)

        # Can fit from restored model or train new model
        if not self.is_instantiated:
            self._instantiate_model(sess, train_ratings)

        n_users = eval_ratings.shape[0]
        n_users_eval = min(n_users, self.n_users_eval)

        print('CREATING FEEDER')
        feeder = self._create_feeder(train_ratings)

        print('START TRAINING')
        f = open(self.model_dir + '/metrics.txt', 'w')
        f.close()

        best_score = 0

        for i in range(self.n_epochs):
            losses = []
            for f_dict in tqdm(feeder.generate_feeds(),
                               desc='Optimizing..., epoch {}/{}'.format(i+1, self.n_epochs)):
                _, loss = sess.run([self.train_ops, self.loss], feed_dict=f_dict)
                losses.append(loss)
            print('Average loss between epoch {} and {}: {}'.format(i, i + 1, np.mean(losses)))

            if i%self.eval_every == 0:
                user_ids = self.state.choice(n_users, n_users_eval,
                                             replace=False)
                pdists = sess.run(self.pdists(user_ids))
                ndcg_at_10 = np.mean(ndcg_at_k(pos_eval_csr[user_ids, :],
                                    taboo_csr[user_ids, :], pdists, 10))
                line = 'Epoch {}, NDCG@10 on eval set: {}'.format(i+1, ndcg_at_10)
                print(line)

                # Update if best
                if ndcg_at_10 > best_score:
                    if save:
                        self.save(sess)
                    best_score = ndcg_at_10
                    f = open(self.model_dir + '/metrics.txt', 'w')
                    f.write('NDCG@10 on eval set: {}'.format(best_score))
                    f.close()

        print('\n\n')

    def save(self, sess):
        print('SAVING MODEL IN DIRECTORY: {}'.format(self.model_dir))
        np.save(self.model_dir + '/users', sess.run(self.users))
        np.save(self.model_dir + '/items', sess.run(self.items))

    def _instantiate_model(self, sess, train_data):

        self._initialize_variables(sess, train_data)

        # instantiate computational graph and train ops
        print('COMPUTING GRAPH AND TRAIN OPS')
        self._create_graph()

        # Run initialization
        sess.run(tf.global_variables_initializer())

        self.is_instantiated = True

    def _initialize_variables(self, sess, train_data):
        if train_data is not None:
            m, n = train_data.shape
            self.users = tf.get_variable('users', shape=(m, self.r),
                                         initializer=tf.random_normal_initializer())
            self.items = tf.get_variable('items', shape=(n, self.r),
                                         initializer=tf.random_normal_initializer())
        else:
            embeddings_dir = self.model_dir
            users_np = np.load(embeddings_dir + '/users.npy')
            items_np = np.load(embeddings_dir + '/items.npy')
            self.users = tf.get_variable('users', initializer=users_np)
            self.items = tf.get_variable('items', initializer=items_np)

    def _gen_dir_name(self):
        raise NotImplementedError('_gen_dir_name() method should be implemented'
                                  'in a concrete model')

    def _create_graph(self):
        raise NotImplementedError('_create_graph() method should be '
                                  'implemented in a concrete model')

    def _create_feeder(self, train_data):
        raise NotImplementedError('_create_feeder() method should be '
                                  'implemented in a concrete model')

class Feeder(object):

    def __init__(self, train_data, *args, **kwargs):
        pass

    def generate_feeds(self):
        yield None

if __name__ == '__main__':
    pass
