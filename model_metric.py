# Custom F1_score for keras
# keeping track of true positives, predicted positives, and all possible positives throughout the whole epoch and then calculating the f1 score at the end of the epoch
# (NOT only giving the f1 score for each batch which isn't really the best metric when we really want the f1 score of the all the data)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models


def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred>=0.5, 1., 0.)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function

class F1_score(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight('all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight('all_possible_positives', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)

    def result(self):
        precision = self.tp_count / self.all_predicted_positives
        recall = self.tp_count / self.all_possible_positives
        f1 = 2*(precision*recall)/(precision+recall)
        return f1

bi_lstm = models.load_model('bi_lstm.h5',
                            custom_objects={'F1_score':F1_score})
    
bi_lstm.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=[F1_score()])

labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
            'excitement', 'fear', 'sadness', 'something else']