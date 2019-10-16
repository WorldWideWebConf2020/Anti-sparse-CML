import numpy as np
import tensorflow as tf
from models import Model
from models.Pointwise import ChFeeder
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group
from models import Feeder
import os

class CH(Model):

    def __init__(self, params):
        self.name = 'ch'
        self.lambd = params['lambda']
        super(CH, self).__init__(params)
        # Compute model directory from params
        dir_name = self._gen_dir_name()
        self.model_dir = os.path.join(self.saving_dir, self.name,
                                    'r={}'.format(self.r), dir_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _gen_dir_name(self):
        str_params = 'lambda={},'.format(self.lambd)
        str_params += 'n_epochs={}'.format(self.n_epochs)
        return str_params

    def _initialize_variables(self, sess, train_data):
        if train_data is not None:
            print('INITIALIZING VARIABLES')
            self.n, self.m = train_data.shape
            self.mult = self.lambd/(2*self.r*self.m*self.n)

            pca = PCA(n_components=self.r)
            emb_users = pca.fit_transform(train_data.toarray()).astype(np.float32)
            emb_users /= np.sum(emb_users ** 2, axis=0)
            emb_items = pca.fit_transform(train_data.toarray().T).astype(np.float32)
            emb_items /= np.sum(emb_items ** 2, axis=0)

            self.users = tf.get_variable('users', initializer=emb_users.T,
                                         dtype=tf.float32, trainable=False)
            self.items = tf.get_variable('items', initializer=emb_items.T,
                                         dtype=tf.float32, trainable=False)

            self.h_users = tf.get_variable('h_users', initializer=np.sign(emb_users.T),
                                           dtype=tf.float32, trainable=False)
            self.h_items = tf.get_variable('h_items', initializer=np.sign(emb_items.T),
                                           dtype=tf.float32, trainable=False)

            self.rot_u = tf.get_variable('rot_u',
                                         initializer=special_ortho_group.rvs(self.r).astype(np.float32),
                                         dtype=tf.float32, trainable=False)
            self.rot_i = tf.get_variable('rot_i',
                                         initializer=special_ortho_group.rvs(self.r).astype(np.float32),
                                         dtype=tf.float32, trainable=False)

            self.sigma = tf.get_variable('sigma', initializer=np.array(5.0).astype(np.float32),
                                         dtype=tf.float32, trainable=False)

        else:
            print('RESTORING MODEL FROM {}'.format(self.model_dir))
            h_users = np.load(self.model_dir + '/h_users.npy')
            h_items = np.load(self.model_dir + '/h_items.npy')
            emb_users = np.load(self.model_dir + '/users.npy')
            emb_items = np.load(self.model_dir + '/items.npy')
            rot_u = np.load(self.model_dir + '/rot_u.npy')
            rot_i = np.load(self.model_dir + '/rot_i.npy')
            sigma = np.load(self.model_dir + '/sigma.npy')
            self.users = tf.get_variable('users', initializer=emb_users)
            self.items = tf.get_variable('items', initializer=emb_items)
            self.rot_u = tf.get_variable('rot_u', initializer=rot_u)
            self.rot_i = tf.get_variable('rot_i', initializer=rot_i)
            self.sigma = tf.get_variable('sigma', initializer=sigma)
            self.h_users = tf.get_variable('h_users', initializer=h_users)
            self.h_items = tf.get_variable('h_items', initializer=h_items)
            self.n, self.m = int(self.users.shape[1]), int(self.items.shape[1])
            self.mult = self.lambd/(2*self.r*self.m*self.n)

        self.J = tf.ones((self.n, self.m), dtype=tf.float32)

    def save(self, sess):
        print('SAVING MODEL IN DIRECTORY: {}'.format(self.model_dir))
        np.save(self.model_dir + '/h_users', sess.run(self.h_users))
        np.save(self.model_dir + '/h_items', sess.run(self.h_items))
        np.save(self.model_dir + '/users', sess.run(self.users))
        np.save(self.model_dir + '/items', sess.run(self.items))
        np.save(self.model_dir + '/rot_u', sess.run(self.rot_u))
        np.save(self.model_dir + '/rot_i', sess.run(self.rot_i))
        np.save(self.model_dir + '/sigma', sess.run(self.sigma))

    def pdists(self, user_ids):
        h_users = tf.gather(tf.transpose(self.h_users), user_ids)
        return -tf.matmul(h_users, self.h_items)

    def _create_feeder(self, train_ratings):
        return ChFeeder(train_ratings, self.ratings_ph)

    def _create_graph(self):

        # Placeholders
        self.ratings_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])

        # Inference
        scaled_ratings = self.ratings_ph - self.sigma*self.J/2

        new_uh = self._hash_updates(self.users, self.h_items,
                                    tf.transpose(scaled_ratings), self.rot_u)
        self.update_uh_op = self.h_users.assign(new_uh)

        with tf.control_dependencies([self.update_uh_op]):
            new_ih = self._hash_updates(self.items, self.h_users,
                                        scaled_ratings, self.rot_i)
            self.update_ih_op = self.h_items.assign(new_ih)

        with tf.control_dependencies([self.update_ih_op]):
            new_rot_u = self._svd(self.d_tb(self.h_users, self.users))
            self.update_rot_u = self.rot_u.assign(new_rot_u)

        with tf.control_dependencies([self.update_rot_u]):
            new_rot_i = self._svd(self.d_tb(self.h_items, self.items))
            self.update_rot_i = self.rot_i.assign(new_rot_i)

        with tf.control_dependencies([self.update_rot_i]):
            M = 0.5*(self.J + self.d_ta(self.h_users, self.h_items)/self.r)
            flat_M = tf.reshape(M, (-1, 1))
            flat_ratings = tf.reshape(self.ratings_ph, (-1, 1))
            new_sigma = self.d_ta(flat_M, flat_ratings)/self.d_ta(flat_M, flat_M)
            # Define final train ops
            self.train_ops = self.sigma.assign(tf.squeeze(new_sigma))

        # Define loss
        self.loss = tf.reduce_mean((self.ratings_ph-self.sigma*M)**2)


    def _hash_updates(self, a, b, scaled_ratings, rot):
        r = tf.matmul(rot, a)
        diff = tf.matmul(b, scaled_ratings)
        D = r/tf.cast(a.shape[1], tf.float32) + self.sigma*self.mult*diff
        _, s, s_tilde = tf.linalg.svd(D)
        return tf.sign(self.d_tb(s, s_tilde))

    def _svd(self, M):
        _, s, s_tilde = tf.linalg.svd(M)
        return self.d_tb(s, s_tilde)

    @staticmethod
    def d_ta(x, y):
        return tf.matmul(x, y, transpose_a=True)

    @staticmethod
    def d_tb(x, y):
        return tf.matmul(x, y, transpose_b=True)
