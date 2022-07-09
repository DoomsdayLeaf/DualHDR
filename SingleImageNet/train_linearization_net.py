import logging

logging.basicConfig(level=logging.INFO)
import os
import tensorflow as tf
from SingleImageNet import util, linearization_net
from DataProcess import dataset
from Dual_SingleImageNet import non_linearization_net

# --- graph
class train_linearization_net(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        self._clip = lambda x: tf.clip_by_value(x, 0, 1)

        self.batch_size = args.linear_batch_size
        self.it_num = args.linear_epoch
        self.logdir_path = args.linear_indi_log_dir
        self.hdr_prefix = args.hdr_prefix

        self.dual_model = False

        self.epsilon = 0.001
        self.dual_lamada = 0.3

    def rand_quantize(
            self,
            img,  # [b, h, w, c]
            is_training):
        b, h, w, c, = util.get_tensor_shape(img)

        const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

        bit = tf.cond(is_training, lambda: const_bit, lambda: const_bit)
        s = (2 ** bit) - 1

        img = self._clip(img)
        img = tf.round(s * img) / s
        img = self._clip(img)

        return img

    def build_graph(
            self,
            hdr,  # [b, h, w, c]
            crf,  # [b, k]
            invcrf,
            t,  # [b]
            is_training,
    ):

        b, h, w, c, = util.get_tensor_shape(hdr)
        b, k, = util.get_tensor_shape(crf)
        b, = util.get_tensor_shape(t)

        """ compute standard model """
        self._hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

        # Augment Poisson and Gaussian noise
        sigma_s = 0.08 / 6 * tf.random_uniform([tf.shape(self._hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                               dtype=tf.float32, seed=1)
        sigma_c = 0.005 * tf.random_uniform([tf.shape(self._hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                            dtype=tf.float32, seed=1)
        noise_s_map = sigma_s * self._hdr_t
        noise_s = tf.random_normal(shape=tf.shape(self._hdr_t), seed=1) * noise_s_map
        temp_x = self._hdr_t + noise_s
        noise_c = sigma_c * tf.random_normal(shape=tf.shape(self._hdr_t), seed=1)
        temp_x = temp_x + noise_c
        self._hdr_t = tf.nn.relu(temp_x)

        # Dynamic range clipping
        self.clipped_hdr_t = self._clip(self._hdr_t)

        # Camera response function
        self.ldr = util.apply_rf(self.clipped_hdr_t, crf)

        # Quantization and JPEG compression
        self.quantized_hdr = tf.round(self.ldr * 255.0)
        self.quantized_hdr_8bit = tf.cast(self.quantized_hdr, tf.uint8)
        jpeg_img_list = []
        for i in range(self.batch_size):
            II = self.quantized_hdr_8bit[i]
            II = tf.image.adjust_jpeg_quality(II, int(round(float(i) / float(self.batch_size - 1) * 10.0 + 90.0)))
            jpeg_img_list.append(II)
        self.jpeg_img = tf.stack(jpeg_img_list, 0)
        self.jpeg_img_float = tf.cast(self.jpeg_img, tf.float32) / 255.0
        self.jpeg_img_float.set_shape([None, 256, 256, 3])

        # loss mask to exclude over-/under-exposed regions
        gray = tf.image.rgb_to_grayscale(self.jpeg_img)
        over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
        over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
        over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
        under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
        under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
        under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
        extreme_cases = tf.logical_or(over_exposed, under_exposed)
        self.loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

        """ build primary model """
        self.lin_net = linearization_net.Linearization_net()
        self.pred_invcrf = self.lin_net.get_output(self.ldr, is_training)
        self.pred_lin_ldr = util.apply_rf(self.ldr, self.pred_invcrf)

        """ compute primary loss """
        self.loss = util.get_l2_loss(self.pred_lin_ldr, self.clipped_hdr_t)
        self.final_loss = self.loss


        """ set train opeator """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean((self.final_loss)))

        """ comput psnr """
        mse = tf.reduce_mean((self.pred_lin_ldr - self.clipped_hdr_t) ** 2)
        self.psnr = 20.0 * util.log10(1.0) - 10.0 * util.log10(mse)
        mse = tf.reduce_mean((self.ldr - self.clipped_hdr_t) ** 2)
        self.psnr_no_q = 20.0 * util.log10(1.0) - 10.0 * util.log10(mse)

        """ save debug log """
        self.show_loss = tf.reduce_mean(self.loss)
        self.show_final_loss = tf.reduce_mean(self.final_loss)

        tf.summary.scalar('loss', self.show_loss)
        tf.summary.scalar('final_loss', self.show_final_loss)
        tf.summary.image('pred_lin_ldr', self.pred_lin_ldr)
        tf.summary.image('ldr', self.ldr)
        tf.summary.image('clipped_hdr_t', self.clipped_hdr_t)
        tf.summary.scalar('loss mask 0', tf.squeeze(self.loss_mask[0]))
        tf.summary.scalar('loss mask 1', tf.squeeze(self.loss_mask[1]))
        tf.summary.scalar('loss mask 2', tf.squeeze(self.loss_mask[2]))
        tf.summary.scalar('psnr', self.psnr)
        tf.summary.scalar('psnr_no_q', self.psnr_no_q)


    def train(self):
        b, h, w, c = self.batch_size, 256, 256, 3

        hdr = tf.placeholder(tf.float32, [None, h, w, c])
        crf = tf.placeholder(tf.float32, [None, None])
        invcrf = tf.placeholder(tf.float32, [None, None])
        t = tf.placeholder(tf.float32, [None])
        is_training = tf.placeholder(tf.bool)

        self.build_graph(hdr, crf, invcrf, t, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # ---

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print('total params: ')
        print(total_parameters)

        self.sess.run(tf.global_variables_initializer())
        self.lin_net.crf_feature_net.overwrite_init(self.sess)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir_path, 'summary'),
            self.sess.graph,
        )
        dataset_reader = dataset.RandDatasetReader(dataset.get_train_dataset(self.hdr_prefix), b)

        for it in range(self.it_num):
            print(it)
            if it == 0 or it % 10000 == 9999:
                print('start save')
                checkpoint_path = os.path.join(self.logdir_path, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=it)
                print('finish save')
            hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
            _, summary_val = self.sess.run([self.train_op, summary], {
                hdr: hdr_val,
                crf: crf_val,
                invcrf: invcrf_val,
                t: t_val,
                is_training: True,
            })
            if it == 0 or it % 10000 == 9999:
                summary_writer.add_summary(summary_val, it)
                logging.info('test')
