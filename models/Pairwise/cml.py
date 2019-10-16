import tensorflow as tf
from models.Pairwise import PairWise
import numpy as np
import scipy.io
import os

class CML(PairWise):

    def __init__(self, params):
        self.name = 'cml'
        self.margin = params['margin']
        super(CML, self).__init__(params)
        # Compute model directory from params
        dir_name = self._gen_dir_name()
        self.model_dir = os.path.join(self.saving_dir, self.name,
                                    'r={}'.format(self.r), dir_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

    @staticmethod
    def _distance(x, y):
        batch_mul = tf.matmul(x, tf.transpose(y, (0, 2, 1)))
        norm_x = tf.reduce_sum(x ** 2, axis=-1)
        norm_y = tf.reduce_sum(y ** 2, axis=-1)
        return norm_x - 2 * batch_mul + norm_y

    def pdists(self, user_ids):
        users = tf.gather(self.users, user_ids)
        items = self.items
        dot_prod = tf.matmul(users, items, transpose_b=True)
        norm_users = tf.reduce_sum(users ** 2, axis=1)[:, None]
        norm_items = tf.reduce_sum(items ** 2, axis=1)[None, :]
        return norm_users - 2*dot_prod + norm_items

    def _create_inference(self):
        super(CML, self)._create_inference()
        # Get unique sampled ids
        self.unique_u_ids, _ = tf.unique(self.anchor_ids)
        flat_negs_ids = tf.reshape(self.neg_ids, (-1,))
        self.unique_i_ids, _ = tf.unique(tf.concat([self.pos_ids, flat_negs_ids],
                                         axis=0))

        # Get corresponding embeddings
        self.unique_u = tf.nn.embedding_lookup(self.users, self.unique_u_ids)
        self.unique_i = tf.nn.embedding_lookup(self.items, self.unique_i_ids)


    def _create_loss(self):
        self._embedding_loss = tf.maximum(self._diff() + self.margin, 0.0)
        self.loss = tf.reduce_mean(self._embedding_loss)

    def _create_train_ops(self):
        print('CREATING TRAIN OPS')
        step = [self.optimizer.minimize(self.loss)]
        with tf.control_dependencies(step):
            self.train_ops = self._projection_op()

    def _projection_op(self):

        # Clip norms
        clipped_u = tf.clip_by_norm(self.unique_u, 1, axes=1)
        clipped_i = tf.clip_by_norm(self.unique_i, 1, axes=1)
        return [tf.scatter_update(self.users, self.unique_u_ids, clipped_u),
                tf.scatter_update(self.items, self.unique_i_ids, clipped_i)]

    def _gen_dir_name(self):
        str_params = 'margin={},'.format(self.margin)
        str_params += 'b_size={},'.format(self.b_size)
        str_params += 'lr={},'.format(self.lr)
        str_params += 'n_negs={},'.format(self.n_negatives)
        str_params += 'n_eps={},'.format(self.n_epochs)
        str_params += 'seed={}'.format(self.seed)
        return str_params

class AntiSparseCML(CML):

    def __init__(self, params):
        self.name = 'anti_sparse_cml'
        self.lambd = params['lambda']
        super(AntiSparseCML, self).__init__(params)
        self.name = 'anti_sparse_cml'
        # Compute model directory from params
        dir_name = self._gen_dir_name()
        self.model_dir = os.path.join(self.saving_dir, self.name,
                                    'r={}'.format(self.r), dir_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _create_loss(self):
        super(AntiSparseCML, self)._create_loss()
        norm_anchors = tf.reduce_sum(tf.abs(self.anchors), axis=1)
        norm_pos = tf.reduce_sum(tf.abs(self.positives), axis=1)
        flat_negs = tf.reshape(self.negatives, (-1, self.r))
        norm_negs = tf.reduce_sum(tf.abs(flat_negs), axis=1)
        self.loss -= (tf.reduce_mean(norm_anchors)/2
                    + tf.reduce_mean(norm_pos)/4
                    + tf.reduce_mean(norm_negs)/4)*self.lambd

    def _gen_dir_name(self):
        str_params = super(AntiSparseCML, self)._gen_dir_name()
        return 'lambda={},'.format(self.lambd) + str_params

    def pdists(self, user_ids):
        users = tf.sign(tf.gather(self.users, user_ids))
        items = tf.sign(self.items)
        dot_prod = tf.matmul(users, items, transpose_b=True)
        return -dot_prod

class SignCML(CML):

    def __init__(self, params):
        self.name = 'sign_cml'
        super(SignCML, self).__init__(params)
        self.name = 'sign_cml'
        # Compute model directory from params
        dir_name = self._gen_dir_name()
        self.model_dir = os.path.join(self.saving_dir, self.name,
                                    'r={}'.format(self.r), dir_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def pdists(self, user_ids):
        users = tf.sign(tf.gather(self.users, user_ids))
        items = tf.sign(self.items)
        dot_prod = tf.matmul(users, items, transpose_b=True)
        return -dot_prod
