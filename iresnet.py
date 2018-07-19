import tensorflow as tf
import sys

def tf_warp_img(im, disp):
    b = tf.shape(im)[0]
    h = tf.shape(im)[1]
    w = tf.shape(im)[2]
    c = tf.shape(im)[3]

    disp = tf.squeeze(disp)

    def _warp(i):
        x, y = tf.meshgrid(tf.range(w), tf.range(h))
        x_f = tf.to_float(x)
        x_f -= disp[i]
        x0_f = tf.floor(x_f)
        x1_f = x0_f + 1

        w0 = x1_f - x_f
        w0 = tf.expand_dims(w0, axis=2)
        w1 = x_f - x0_f
        w1 = tf.expand_dims(w1, axis=2)

        x_0 = tf.zeros(shape=[h, w], dtype=tf.float32)
        x_w = tf.ones(shape=[h, w], dtype=tf.float32) * tf.to_float(w - 1)
        x0_f = tf.where(x0_f < 0, x_0, x0_f)
        x0_f = tf.where(x0_f > tf.to_float(w - 1), x_w, x0_f)
        x1_f = tf.where(x1_f < 0, x_0, x1_f)
        x1_f = tf.where(x1_f > tf.to_float(w - 1), x_w, x1_f)

        x0_f = tf.expand_dims(x0_f, axis=2)
        x1_f = tf.expand_dims(x1_f, axis=2)
        y = tf.expand_dims(y, axis=2)
        indices = tf.concat([y, tf.to_int32(x0_f)], axis=2)
        indices = tf.reshape(indices, [-1, 2])
        iml = tf.gather_nd(im[i], indices)
        indices = tf.concat([y, tf.to_int32(x1_f)], axis=2)
        indices = tf.reshape(indices, [-1, 2])
        imr = tf.gather_nd(im[i], indices)

        res = w0 * tf.reshape(iml, [h, w, c]) + w1 * tf.reshape(imr, [h, w, c])
        return res

    ret = tf.map_fn(_warp, tf.range(b), dtype=tf.float32)
    ret = tf.reshape(ret, [b, h, w, c])
    return ret

def correlation_map(x, y, max_disp, name):
    w = tf.shape(y)[2]
    corr_tensors = []
    for i in range(-max_disp, 0, 1):
        shifted = tf.pad(y[:, :, 0:w+i, :], [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
        corr_tensors.append(corr)
    for i in range(max_disp + 1):
        shifted = tf.pad(x[:, :, i:, :], [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
        corr_tensors.append(corr)
    return tf.transpose(tf.stack(corr_tensors), perm=[1, 2, 3, 0])


class IResNet(object):
    def __init__(self, max_disp=40, mode='inference', corr_type='tf'):

        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
        self.iml = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_left')
        self.imr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_right')
        self.max_disp = max_disp
        self.mode = mode
        self.corr_type = corr_type
        self.disp = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='disp')
        self.weight_decay = tf.placeholder(tf.float32, shape=[], name='weight_decay')
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
        self.ires_disp_loss1 = None
        self.ires_disp_loss2 = None
        self.loss_decay = None
        self.pred = None
        self.error = None
        self.total_loss = None
        self.build_train_model()
        if self.mode=='train':
            self.build_summary()

    def conv2d_relu(self, inp, filters, kernel_size, stride, name, kernel_regularizer=None, reuse=None):
        return tf.layers.conv2d(inp, filters, kernel_size, stride, 'SAME',
                                kernel_regularizer=kernel_regularizer, activation=tf.nn.relu, name=name, reuse=reuse)

    def conv2d(self, inp, filters, kernel_size, stride, name, kernel_regularizer=None, reuse=None):
        return tf.layers.conv2d(inp, filters, kernel_size, stride, 'SAME',
                                kernel_regularizer=kernel_regularizer, name=name, reuse=reuse)

    def deconv2d_relu(self, inp, filters, kernel_size, stride, name, kernel_regularizer=None, reuse=None):
        return tf.layers.conv2d_transpose(inp, filters, kernel_size, stride, 'SAME',
                                          kernel_regularizer=kernel_regularizer, activation=tf.nn.relu, name=name, reuse=reuse)

    def deconv2d(self, inp, filters, kernel_size, stride, name, kernel_regularizer=None, reuse=None):
        return tf.layers.conv2d_transpose(inp, filters, kernel_size, stride, 'SAME',
                                          kernel_regularizer=kernel_regularizer, name=name, reuse=reuse)

    def build_train_model(self):

        with tf.name_scope('feature_extraction'):
            conv1a = self.conv2d_relu(self.iml, 64, 7, 2, name='conv1', kernel_regularizer=self.kernel_regularizer)
            conv1b = self.conv2d_relu(self.imr, 64, 7, 2, name='conv1', kernel_regularizer=self.kernel_regularizer, reuse=True)

            conv2a = self.conv2d_relu(conv1a, 128, 5, 2, name='conv2', kernel_regularizer=self.kernel_regularizer)
            conv2b = self.conv2d_relu(conv1b, 128, 5, 2, name='conv2', kernel_regularizer=self.kernel_regularizer, reuse=True)

            corr_conv3a = self.conv2d_relu(conv2a, 256, 3, 2, name='corr_conv3', kernel_regularizer=self.kernel_regularizer)
            corr_conv3b = self.conv2d_relu(conv2b, 256, 3, 2, name='corr_conv3', kernel_regularizer=self.kernel_regularizer, reuse=True)

            corr_conv3_1a = self.conv2d_relu(corr_conv3a, 256, 3, 1, name='corr_conv3_1')
            corr_conv3_1b = self.conv2d_relu(corr_conv3b, 256, 3, 1, name='corr_conv3_1', reuse=True)

            corr_deconv3a = self.deconv2d_relu(corr_conv3_1a, 128, 4, 2, name='corr_deconv3')
            corr_deconv3b = self.deconv2d_relu(corr_conv3_1b, 128, 4, 2, name='corr_deconv3', reuse=True)

            edcorr_2a = self.conv2d_relu(tf.concat([corr_deconv3a, conv2a], axis=-1), 128, 3, 1, name='corr_fusion2')
            edcorr_2b = self.conv2d_relu(tf.concat([corr_deconv3b, conv2b], axis=-1), 128, 3, 1, name='corr_fusion2', reuse=True)

            corr = correlation_map(edcorr_2a, edcorr_2b, max_disp=50, name='corr')
            corr.set_shape([None, None, None, 101])

            conv_redir = self.conv2d_relu(conv2a, 64, 1, 1, name='conv_redir', kernel_regularizer=self.kernel_regularizer)
            conv3 = self.conv2d_relu(tf.concat([corr, conv_redir], axis=-1), 256, 5, 2, name='conv3', kernel_regularizer=self.kernel_regularizer)
            conv3_1 = self.conv2d_relu(conv3, 256, 3, 1, name='conv3_1', kernel_regularizer=self.kernel_regularizer)
            conv4 = self.conv2d_relu(conv3_1, 512, 3, 2, name='conv4', kernel_regularizer=self.kernel_regularizer)
            conv4_1 = self.conv2d_relu(conv4, 512, 3, 1, name='conv4_1', kernel_regularizer=self.kernel_regularizer)
            conv5 = self.conv2d_relu(conv4_1, 512, 3, 2, name='conv5', kernel_regularizer=self.kernel_regularizer)
            conv5_1 = self.conv2d_relu(conv5, 512, 3, 1, name='conv5_1', kernel_regularizer=self.kernel_regularizer)
            conv6 = self.conv2d_relu(conv5_1, 1024, 3, 2, name='conv6', kernel_regularizer=self.kernel_regularizer)
            conv6_1 = self.conv2d_relu(conv6, 1024, 3, 1, name='conv6_1', kernel_regularizer=self.kernel_regularizer)

        with tf.name_scope('disparity_regression'):
            disp6 = self.conv2d_relu(conv6_1, 1, 3, 1, name='predict_flow6')

            uconv5 = self.deconv2d_relu(conv6_1, 512, 4, 2, name='uconv5')
            disp6_up = self.deconv2d(disp6, 1, 4, 2, name='disp6_up')
            iconv5 = self.conv2d(tf.concat([conv5_1, uconv5, disp6_up], axis=-1), 512, 3, 1, name='iconv5')
            disp5 = self.conv2d_relu(iconv5, 1, 3, 1, name='disp5')

            uconv4 = self.deconv2d_relu(iconv5, 256, 4, 2, name='uconv4')
            disp5_up = self.deconv2d(disp5, 1, 4, 2, name='disp5_up')
            iconv4 = self.conv2d(tf.concat([conv4_1, uconv4, disp5_up], axis=-1), 256, 3, 1, name='iconv4')
            disp4 = self.conv2d_relu(iconv4, 1, 3, 1, name='disp4')

            uconv3 = self.deconv2d_relu(iconv4, 128, 4, 2, name='uconv3')
            disp4_up = self.deconv2d(disp4, 1, 4, 2, name='disp4_up')
            iconv3 = self.conv2d(tf.concat([conv3_1, uconv3, disp4_up], axis=-1), 128, 3, 1, name='iconv3')
            disp3 = self.conv2d_relu(iconv3, 1, 3, 1, name='disp3')

            uconv2 = self.deconv2d_relu(iconv3, 64, 4, 2, name='uconv2')
            disp3_up = self.deconv2d(disp3, 1, 4, 2, name='disp3_up')
            iconv2 = self.conv2d(tf.concat([edcorr_2a, uconv2, disp3_up], axis=-1), 64, 3, 1, name='iconv2')
            disp2 = self.conv2d_relu(iconv2, 1, 3, 1, name='disp2')

            uconv1 = self.deconv2d_relu(iconv2, 32, 4, 2, name='uconv1')
            disp2_up = self.deconv2d(disp2, 1, 4, 2, name='disp2_up')
            iconv1 = self.conv2d(tf.concat([conv1a, uconv1, disp2_up], axis=-1), 32, 3, 1, name='iconv1')
            disp1 = self.conv2d_relu(iconv1, 1, 3, 1, name='disp1')

        with tf.name_scope('multi_scale_full_disparity'):
            up_conv1a = self.deconv2d_relu(conv1a, 32, 4, 2, name='up_conv1ab')
            up_conv1b = self.deconv2d_relu(conv1b, 32, 4, 2, name='up_conv1ab', reuse=True)

            up_conv2a = self.deconv2d_relu(conv2a, 32, 8, 4, name='up_conv2ab')
            up_conv2b = self.deconv2d_relu(conv2b, 32, 8, 4, name='up_conv2ab', reuse=True)

            up_conv1a2a = self.conv2d_relu(tf.concat([up_conv1a, up_conv2a], axis=-1), 32, 1, 1, name='up_conv1a2a',
                                           kernel_regularizer=self.kernel_regularizer)
            up_conv1b2b = self.conv2d_relu(tf.concat([up_conv1b, up_conv2b], axis=-1), 32, 1, 1, name='up_conv1b2b',
                                           kernel_regularizer=self.kernel_regularizer)

            uconv0 = self.deconv2d_relu(iconv1, 32, 4, 2, name='uconv0')
            disp1_up = self.deconv2d(disp1, 1, 4, 2, name='disp1_up')
            iconv0 = self.conv2d(tf.concat([up_conv1a2a, uconv0, disp1_up], axis=-1), 32, 3, 1, name='iconv0')
            disp0 = self.conv2d_relu(iconv0, 1, 3, 1, name='disp0')

            subupsample_disp6 = self.deconv2d(disp6, 1, 128, 64, name='subupsample_disp6')
            subupsample_disp5 = self.deconv2d(disp5, 1, 64, 32, name='subupsample_disp5')
            subupsample_disp4 = self.deconv2d(disp4, 1, 32, 16, name='subupsample_disp4')
            subupsample_disp3 = self.deconv2d(disp3, 1, 16, 8, name='subupsample_disp3')
            subupsample_disp2 = self.deconv2d(disp2, 1, 8, 4, name='subupsample_disp2')
            subupsample_disp1 = self.deconv2d(disp1, 1, 4, 2, name='subupsample_disp1')

            multi_res_prediction = tf.concat([subupsample_disp6, subupsample_disp5,
                                              subupsample_disp4, subupsample_disp3,
                                              subupsample_disp2, subupsample_disp1, disp0], axis=-1)

            final_prediction = self.conv2d_relu(multi_res_prediction, 1, 1, 1, name='predict_from_multi_res')

        with tf.name_scope('refinement'):
            conv1a_mini = self.conv2d_relu(conv1a, 16, 3, 1, 'compress_conv1a1b', kernel_regularizer=self.kernel_regularizer)
            conv1b_mini = self.conv2d_relu(conv1b, 16, 3, 1, 'compress_conv1a1b', kernel_regularizer=self.kernel_regularizer, reuse=True)
            corr_mini = correlation_map(conv1a_mini, conv1b_mini, max_disp=40, name='corr_mini')
            corr_mini.set_shape([None, None, None, 81])

            w_up_1b2b = tf_warp_img(up_conv1b2b, final_prediction)

            ires_conv0_itr1 = self.conv2d_relu(tf.concat([up_conv1a2a, tf.abs(up_conv1a2a - w_up_1b2b), final_prediction], axis=-1),
                                               32, 3, 1, 'ires_conv0_itr1', kernel_regularizer=self.kernel_regularizer)
            ires_conv1_itr1 = self.conv2d_relu(ires_conv0_itr1, 64, 3, 2, 'ires_conv1_itr1', kernel_regularizer=self.kernel_regularizer)

            ires_conv1b_itr1 = self.conv2d_relu(tf.concat([corr_mini, ires_conv1_itr1], axis=-1),
                                                64, 3, 1, name='ires_conv1b_itr1', kernel_regularizer=self.kernel_regularizer)
            ires_conv2_itr1 = self.conv2d_relu(ires_conv1b_itr1, 128, 3, 2, name='ires_conv2_itr1', kernel_regularizer=self.kernel_regularizer)
            ires_conv2b_itr1 = self.conv2d_relu(ires_conv2_itr1, 128, 3, 1, name='r_conv2_1', kernel_regularizer=self.kernel_regularizer)

            ires_predict2_res_itr1 = self.conv2d(ires_conv2b_itr1, 1, 3, 1, name='ires_predict2_res_itr1')
            if self.mode == 'train':
                ires_predict2_res_itr1_shape = tf.shape(ires_predict2_res_itr1)
                ires_initial2_itr1 = tf.image.resize_images(final_prediction, ires_predict2_res_itr1_shape[1:3])
                ires_predict2_itr1 = ires_initial2_itr1 + ires_predict2_res_itr1
                tgt = tf.image.resize_images(self.disp, ires_predict2_res_itr1_shape[1:3])
                self.ires_disp_loss2 = tf.reduce_mean(tf.abs(ires_predict2_itr1 - tgt))

            ires_deconv2_itr1 = self.deconv2d_relu(ires_conv2b_itr1, 64, 4, 2, name='ires_deconv2_itr1')
            ires_upsampled_2to1_itr1 = self.deconv2d(ires_predict2_res_itr1, 1, 4, 2, name='ires_upsampled_2to1_itr1')
            ires_fused1_itr1 = self.conv2d(tf.concat([ires_conv1b_itr1, ires_deconv2_itr1, ires_upsampled_2to1_itr1], axis=-1), 64, 3, 1, name='ires_fused1_itr1')
            ires_predict1_res_itr1 = self.conv2d(ires_fused1_itr1, 1, 3, 1, name='ires_predict1_res_itr1')
            if self.mode == 'train':
                ires_predict1_res_itr1_shape = tf.shape(ires_predict1_res_itr1)
                ires_initial1_itr1 = tf.image.resize_images(final_prediction, ires_predict1_res_itr1_shape[1:3])
                ires_predict1_itr1 = ires_initial1_itr1 + ires_predict1_res_itr1
                tgt = tf.image.resize_images(self.disp, ires_predict1_res_itr1_shape[1:3])
                self.ires_disp_loss1 = tf.reduce_mean(tf.abs(ires_predict1_itr1 - tgt))

            ires_deconv1_itr1 = self.deconv2d_relu(ires_fused1_itr1, 32, 4, 2, name='ires_deconv1_itr1')
            ires_upsampled_1to0_itr1 = self.deconv2d(ires_predict1_res_itr1, 1, 4, 2, 'ires_upsampled_1to0_itr1')
            ires_fused0_itr1 = self.conv2d(tf.concat([ires_conv0_itr1, ires_deconv1_itr1, ires_upsampled_1to0_itr1], axis=-1), 32, 3, 1, name='r_iconv0')
            ires_predict0_res_itr1 = self.conv2d(ires_fused0_itr1, 1, 3, 1, name='ires_predict0_res_itr1')
            ires_predict0_itr1 = final_prediction + ires_predict0_res_itr1
            self.pred = ires_predict0_itr1
            if self.mode == 'train':
                ires_disp_loss0 = tf.reduce_mean(tf.abs(ires_predict0_itr1 - self.disp))
                self.error = ires_disp_loss0
                self.loss_decay = self.weight_decay * tf.losses.get_regularization_loss()
                self.total_loss = 0.2 * self.ires_disp_loss2 + 0.2 * self.ires_disp_loss1 + 1. * self.error + self.loss_decay
            if self.mode == 'inference':
                final_prediction = ires_predict0_itr1
                w_up_1b2b = tf_warp_img(up_conv1b2b, final_prediction)

                ires_conv0_itr1 = self.conv2d_relu(
                    tf.concat([up_conv1a2a, tf.abs(up_conv1a2a - w_up_1b2b), final_prediction], axis=-1),
                    32, 3, 1, 'ires_conv0_itr1', kernel_regularizer=self.kernel_regularizer, reuse=True)
                ires_conv1_itr1 = self.conv2d_relu(ires_conv0_itr1, 64, 3, 2, 'ires_conv1_itr1',
                                                   kernel_regularizer=self.kernel_regularizer, reuse=True)

                ires_conv1b_itr1 = self.conv2d_relu(tf.concat([corr_mini, ires_conv1_itr1], axis=-1),
                                                    64, 3, 1, name='ires_conv1b_itr1',
                                                    kernel_regularizer=self.kernel_regularizer, reuse=True)
                ires_conv2_itr1 = self.conv2d_relu(ires_conv1b_itr1, 128, 3, 2, name='ires_conv2_itr1',
                                                   kernel_regularizer=self.kernel_regularizer, reuse=True)
                ires_conv2b_itr1 = self.conv2d_relu(ires_conv2_itr1, 128, 3, 1, name='r_conv2_1',
                                                    kernel_regularizer=self.kernel_regularizer, reuse=True)

                ires_predict2_res_itr1 = self.conv2d(ires_conv2b_itr1, 1, 3, 1, name='ires_predict2_res_itr1', reuse=True)

                ires_deconv2_itr1 = self.deconv2d_relu(ires_conv2b_itr1, 64, 4, 2, name='ires_deconv2_itr1', reuse=True)
                ires_upsampled_2to1_itr1 = self.deconv2d(ires_predict2_res_itr1, 1, 4, 2,
                                                         name='ires_upsampled_2to1_itr1', reuse=True)
                ires_fused1_itr1 = self.conv2d(
                    tf.concat([ires_conv1b_itr1, ires_deconv2_itr1, ires_upsampled_2to1_itr1], axis=-1), 64, 3, 1,
                    name='ires_fused1_itr1', reuse=True)
                ires_predict1_res_itr1 = self.conv2d(ires_fused1_itr1, 1, 3, 1, name='ires_predict1_res_itr1', reuse=True)

                ires_deconv1_itr1 = self.deconv2d_relu(ires_fused1_itr1, 32, 4, 2, name='ires_deconv1_itr1', reuse=True)
                ires_upsampled_1to0_itr1 = self.deconv2d(ires_predict1_res_itr1, 1, 4, 2, 'ires_upsampled_1to0_itr1', reuse=True)
                ires_fused0_itr1 = self.conv2d(
                    tf.concat([ires_conv0_itr1, ires_deconv1_itr1, ires_upsampled_1to0_itr1], axis=-1), 32, 3, 1,
                    name='r_iconv0', reuse=True)
                ires_predict0_res_itr1 = self.conv2d(ires_fused0_itr1, 1, 3, 1, name='ires_predict0_res_itr1', reuse=True)
                ires_predict0_itr1 = final_prediction + ires_predict0_res_itr1
                self.pred = ires_predict0_itr1

    def build_summary(self):
        tf.summary.image('oimg_left', self.iml)
        tf.summary.image('oimg_right', self.imr)
        tf.summary.image('disp', self.disp)
        tf.summary.scalar('error', self.error)
        tf.summary.image('pred', self.pred)
        tf.summary.scalar('loss_decay', self.loss_decay)
        tf.summary.scalar('total_loss', self.total_loss)




def test():
    import time
    with tf.device('/cpu:0'):
        iml = np.random.random((3, 256, 256, 3))
        imr = np.random.random((3, 256, 256, 3))
        disp = np.random.random((3, 256, 256, 1))
        net = IResNet(mode='train', corr_type='tf')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = time.time()
        pred, error = sess.run([net.pred, net.error], feed_dict={net.iml: iml,
                                                                 net.imr: imr,
                                                                 net.disp: disp})
        print('iresnet cost {} s'.format(time.time() - start))
        print(pred.shape)
        print(error.shape)


# test()
