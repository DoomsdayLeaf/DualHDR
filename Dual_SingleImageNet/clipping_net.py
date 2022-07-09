import os
import tensorflow as tf
import numpy
from ops import guidedfilter

class Clipping_Net(object):
    def __init__(self):
        self.pix_val_max = 1.0
        self.pix_val_min = 0.0

    def dynamic_range_clipping(self, hdr):
        clip_hdr = tf.clip_by_value(hdr, self.pix_val_min, self.pix_val_max)

        return clip_hdr

    def model(self, hdr):
        ''' base = filter(img) detail = img / base '''
        b = guidedfilter(hdr, 5, 0.01)  # base layer
        d = tf.div(hdr, b + 1e-15)  # detail layer

        # clipped
        b = self.dynamic_range_clipping(b)

        # get final img
        out = b * d

        return out, b, d
