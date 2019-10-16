from models import Feeder
import numpy as np
import tensorflow as tf


class ChFeeder(Feeder):

    def __init__(self, train_data, ratings_ph):
        super(ChFeeder, self).__init__(train_data, {})
        self.train_ratings = train_data.toarray()
        self.ratings_ph = ratings_ph

    def generate_feeds(self):
        return [{self.ratings_ph: self.train_ratings}]
