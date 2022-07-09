from utils import *
import scipy.stats as st


##################################################################################
# Layers
##################################################################################

# convolution layer for JSInet
def conv2d(x, shape, name):
    w = tf.get_variable(name + '/w', shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name + '/b', shape[3], initializer=tf.constant_initializer(0))
    n = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name) + b
    return n


# convolution layer for discriminator
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    weight_regularizer_fully = tf.contrib.layers.l2_regularizer(0.0001)
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x

def pool(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')

def tf_normalize(x):
    max_val = tf.reduce_max(x)
    min_val = tf.reduce_min(x)
    x = (x - min_val) / (max_val - min_val)
    return x

##################################################################################
# Blocks
##################################################################################
# residual block
def res_block(x, c, name):
    with tf.variable_scope(name):
        n = conv2d(relu(x), [3, 3, c, c], 'conv/0')
        n = conv2d(relu(n), [3, 3, c, c], 'conv/1')
        n = x + n
    return n

# EP block
def EP_block(x, name="EP", is_training=True):
    with tf.variable_scope(name):
        ch = 64

        n0 = conv2d(x, [3, 3, x.shape[3], ch], 'conv0')
        n = conv2d(x, [3, 3, x.shape[3], ch], 'conv0/0')
        for i in range(1, 5):
            n = conv2d(relu(n), [3, 3, ch, ch], 'conv' + str(i) + '/0')
            n = conv2d(relu(n), [3, 3, ch, ch], 'conv' + str(i) + '/1')
            n = n + n0

        n = conv2d(n, [3, 3, ch, x.shape[3]], 'conv5')

    return n

# residual block with concat
def res_block_concat(x, c1, c, name):
    with tf.variable_scope(name):
        n = conv2d(relu(x), [3, 3, c1, c], 'conv/0')
        n = conv2d(relu(n), [3, 3, c, c], 'conv/1')
        n = x[:, :, :, :c] + n
    return n


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)

##################################################################################
# Normalization
##################################################################################

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))
    return loss


def discriminator_loss(Ra, real, fake, mode='Hinge'):
    if mode == 'RaHinge':
        # Hinge GAN loss
        real_loss = 0
        fake_loss = 0
        if Ra:
            real_logit = (real - tf.reduce_mean(fake))
            fake_logit = (fake - tf.reduce_mean(real))

            real_loss = tf.reduce_mean(relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))
        else:
            real_loss = tf.reduce_mean(relu(1.0 - real))
            fake_loss = tf.reduce_mean(relu(1.0 + fake))
        loss = real_loss + fake_loss

    elif mode == 'WGAN-GP':
        # WGAN-GP loss
        fake_loss = tf.reduce_mean(fake)
        real_loss = tf.reduce_mean(real)
        loss = fake_loss - real_loss        

    
    return loss


def generator_loss(Ra, real, fake, mode='RaHinge'):
    if mode == 'RaHinge':
        # Hinge GAN loss
        fake_loss = 0
        real_loss = 0
        if Ra:
            fake_logit = (fake - tf.reduce_mean(real))
            real_logit = (real - tf.reduce_mean(fake))

            fake_loss = tf.reduce_mean(relu(1.0 - fake_logit))
            real_loss = tf.reduce_mean(relu(1.0 + real_logit))
        else:
            fake_loss = -tf.reduce_mean(fake)
        loss = fake_loss + real_loss

    elif mode == 'WGAN-GP':
        fake_loss = -tf.reduce_mean(fake)
        loss = fake_loss

    return loss

def get_blend_mask(y, thr, sh, sw):
    mask = tf.reduce_max(y, reduction_indices=[3])
    mask = tf.minimum(1.0, tf.maximum(0.0, mask - 1.0 + thr) / thr)
    mask = tf.reshape(mask, [-1, sh, sw, 1])
    mask = tf.tile(mask, [1, 1, 1, 3])

    return mask

def IR_loss(x, pred, y):
    sb, sy, sx, sc = x.get_shape().as_list()
    eps = 1.0/255.0
    lambda_ir = 0.5

    # For masked loss, only using information near saturated image regions
    thr = 0.05 # Threshold for blending
    msk = get_blend_mask(tf_normalize(y), 0.05, sy, sx)
    
    # Loss separated into illumination and reflectance terms
    if True:
        y_log = tf.log(y+eps)
        pred_log = tf.log(pred + eps)
        pred_ori = tf.exp(pred_log) - eps
        #x_log = tf.log(x * 255+eps)

        # Luminance
        lum_kernel = np.zeros((1, 1, 3, 1))
        lum_kernel[:, :, 0, 0] = 0.213
        lum_kernel[:, :, 1, 0] = 0.715
        lum_kernel[:, :, 2, 0] = 0.072
        y_lum_lin = tf.nn.conv2d(y, lum_kernel, [1, 1, 1, 1], padding='SAME')
        pred_lum_lin = tf.nn.conv2d(pred_ori, lum_kernel, [1, 1, 1, 1], padding='SAME')
        x_lum_lin = tf.nn.conv2d(x, lum_kernel, [1, 1, 1, 1], padding='SAME')

        # Log luminance
        y_lum = tf.log(y_lum_lin + eps)
        pred_lum = tf.log(pred_lum_lin + eps)
        x_lum = tf.log(x_lum_lin + eps)

        # Gaussian kernel
        nsig = 2
        filter_size = 13
        interval = (2*nsig+1.)/(filter_size)
        ll = np.linspace(-nsig-interval/2., nsig+interval/2., filter_size+1)
        kern1d = np.diff(st.norm.cdf(ll))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()

        # Illumination, approximated by means of Gaussian filtering
        weights_g = np.zeros((filter_size, filter_size, 1, 1))
        weights_g[:, :, 0, 0] = kernel
        y_ill = tf.nn.conv2d(y_lum, weights_g, [1, 1, 1, 1], padding='SAME')
        pred_ill = tf.nn.conv2d(pred_lum, weights_g, [1, 1, 1, 1], padding='SAME')
        x_ill = tf.nn.conv2d(x_lum, weights_g, [1, 1, 1, 1], padding='SAME')

        # Reflectance
        y_refl = y_log - tf.tile(y_ill, [1,1,1,3])
        pred_refl = pred_log - tf.tile(pred_ill, [1,1,1,3])
        x_refl = x - tf.tile(x_ill, [1,1,1,3])

        cost =              tf.reduce_mean( ( lambda_ir*tf.square( tf.subtract(pred_ill, y_ill) ) + (1.0-lambda_ir)*tf.square( tf.subtract(pred_refl, y_refl) ) )*msk )
        cost_input_output = tf.reduce_mean( ( lambda_ir*tf.square( tf.subtract(x_ill, y_ill) ) + (1.0-lambda_ir)*tf.square( tf.subtract(x_refl, y_refl) ) )*msk )
    else:
        cost =              tf.reduce_mean( tf.square( tf.subtract(pred, tf.log(y+eps) )*msk ) )
        cost_input_output = tf.reduce_mean( tf.square( tf.subtract(tf.log(y+eps), tf.log(tf.pow(x, 2.0)+eps) )*msk ) );
        
    return cost, cost_input_output, msk

##################################################################################
# Filter
##################################################################################

# guided filter
def guidedfilter(img, r, eps):
    img2 = tf.concat([img, img * img], axis=3)
    img2 = boxfilter(img2, r)
    mean_i, mean_ii = tf.split(img2, 2, axis=3)

    var_i = mean_ii - mean_i * mean_i

    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i

    ab = tf.concat([a, b], axis=3)
    ab = boxfilter(ab, r)

    mean_a, mean_b = tf.split(ab, 2, axis=3)
    q = mean_a * img + mean_b
    return q


def boxfilter(x, szf):
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = tf.ones([szf, szf, 1, 1], tf.float32) / (szf ** 2)
    bf = tf.tile(bf, [1, 1, szy[3], 1])
    pp = int((szf - 1) / 2)

    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')
    return y


##################################################################################
# Misc
##################################################################################

# initialize the uninitialized variables
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

        print("Success to initialize uninitialized variables.")