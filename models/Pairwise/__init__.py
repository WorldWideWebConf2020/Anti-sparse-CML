import tensorflow as tf
import numpy as np
from models import Model
from models import Feeder
import ray
import psutil

class PairWise(Model):

    def __init__(self, params):
        self.n_negatives = params['n_negatives']
        self.b_size = params['batch_size']
        self.lr = params['lr']
        super(PairWise, self).__init__(params)

    def _create_placeholders(self):

        self.anchor_ids = tf.placeholder(name='anchor_ids', dtype=tf.int32,
                                         shape=[None])
        self.pos_ids = tf.placeholder(name='pos_ids', dtype=tf.int32,
                                      shape=[None])
        self.neg_ids = tf.placeholder(name='neg_ids', dtype=tf.int32,
                                      shape=[None, self.n_negatives])

    def _create_inference(self):

        # anchor user embedding: shape=(N, 1, K)
        anchors = tf.nn.embedding_lookup(self.users,
                                         self.anchor_ids,
                                         name='batch_anchor_embeddings')
        self.anchors = tf.expand_dims(anchors, 1)
        # positive item embedding: shape=(N, 1, K)
        positives = tf.nn.embedding_lookup(
            self.items,
            self.pos_ids,
            name='batch_positive_embeddings')
        self.positives = tf.expand_dims(positives, 1)
        # negative item embedding: shape=(N, W, K)
        self.negatives = tf.nn.embedding_lookup(self.items, self.neg_ids,
                                           name='batch_negative_embeddings')

    def _pos_dist(self):
        return self._distance(self.anchors, self.positives)

    def _neg_dist(self):
        return self._distance(self.anchors, self.negatives)

    def _diff(self):
        return self._pos_dist() - self._neg_dist()

    def _create_graph(self):
        self._create_placeholders()
        self._create_inference()
        self._create_loss()
        self._create_train_ops()

    def _create_feeder(self, train_data):
        return TripletFeeder(train_data, self.anchor_ids, self.pos_ids,
                             self.neg_ids, self.params)

    def _create_loss(self):
        raise NotImplementedError('_create_loss() method should be '
                                  'implemented in concrete model')

    def _create_train_ops(self):
        raise NotImplementedError('_create_train_ops() method should be '
                                  'implemented in concrete model')

    def _create_process_embeddings(self):
        raise NotImplementedError('_create_process_embeddings() method should be '
                                  'implemented in concrete model')


class TripletFeeder(Feeder):

    def __init__(self, train_ratings, anchors_ph, pos_ph, neg_ph, params):
        super(TripletFeeder, self).__init__(train_ratings, params)
        pos_train_inters = (train_ratings >=4)
        # Get params for triplet sampling
        self.b_size = params['batch_size']
        self.n_negs = params['n_negatives']
        self.n_epochs = params['n_epochs']

        # Load positive pairs
        list_non_zero_idx = np.split(pos_train_inters.indices, pos_train_inters.indptr)[1:-1]
        u_range = np.arange(pos_train_inters.shape[0])
        i_range = np.arange(pos_train_inters.shape[1])
        self.u_idx, self.i_idx = pos_train_inters.nonzero()
        _, self.counts = np.unique(self.u_idx, return_counts=True)
        self.cum_counts = np.concatenate([np.arange(c) for c in self.counts],
                                          axis=0)

        # Ray pipeline
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)

        # shared mem
        i_range_r = ray.put(i_range)
        inters_r = ray.put(list_non_zero_idx)
        counts_r = ray.put(self.counts)
        n_negs_r = ray.put(self.n_negs)
        n_epochs_r = ray.put(self.n_epochs)

        args = [i_range_r, inters_r, counts_r, n_negs_r, n_epochs_r]
        chunks = np.array_split(u_range, num_cpus)

        # sample all train inputs
        print('SAMPLING TRAINING INPUTS')
        self.neg_sampled = ray_neg_sampling(ray, chunks, *args)
        ray.shutdown()

        # Get place holders
        self.a_ph = anchors_ph
        self.p_ph = pos_ph
        self.n_ph = neg_ph

        self.epoch = 0
        self.n_samples = len(self.u_idx)
        self.n_batches = self.n_samples // self.b_size
        self.pair_feeds = np.stack([self.u_idx, self.i_idx, self.cum_counts],
                                    axis=1)
        self._unpack_negs()

    def generate_feeds(self):

        # Sample random pairs
        batches = np.array_split(np.random.permutation(self.n_samples),
                                 self.n_batches)
        feeds_ = []
        for batch in batches:
            a_ids = self.pair_feeds[batch, 0]
            p_ids = self.pair_feeds[batch, 1]
            n_ids = self.neg_feeds[batch, self.epoch, :]
            feeds_.append({self.a_ph: a_ids, self.p_ph: p_ids,
                   self.n_ph: n_ids})
        self.epoch += 1
        return feeds_

    def _unpack_negs(self):
        l = []
        for u_id, _, c in self.pair_feeds:
            l.append(self.neg_sampled[u_id][c*self.n_epochs*self.n_negs:
                                            (c+1)*self.n_epochs*self.n_negs])
        self.neg_feeds = np.array(l).reshape((self.n_samples, self.n_epochs,
                                              self.n_negs))
        del self.neg_sampled


@ray.remote
def _neg_sampling(u_chunk, i_ids, inters, counts, n_negs, n_epochs):
    a = {}
    for i in u_chunk:
        candidates = np.delete(i_ids, inters[i])
        chosen = np.random.choice(candidates, size=n_negs*n_epochs*counts[i])
        a[i] = chosen
    return a

def ray_neg_sampling(ray, chunk_list, *args):
    d = {}
    res = ray.get([_neg_sampling.remote(chunk, *args) for chunk in chunk_list])
    for d_ in res:
        d.update(d_)
    return d


#@njit
#def jit_neg_sampling(tot_idx, list_non_zero_idx, list_n_samples, n_negatives, n_users):
    #a = {}
    #for i in range(n_users):
        #candidates = np.delete(tot_idx, list_non_zero_idx[i])
        #chosen = np.random.choice(candidates, size=n_negatives*list_n_samples[i])
        #a[i] = chosen
    #return a


if __name__ == '__main__':
    pass
