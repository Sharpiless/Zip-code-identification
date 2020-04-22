import os
import cv2
import tensorflow as tf
from cut_middle import ad_cut_central
from red_check import red_line_check, cut_tiny6, reshape_std
from network import Net
import numpy as np


class ZipId(object):

    def __init__(self):

        self.net = Net(is_training=False)

        self.saver = tf.train.Saver()

    def process_image(self, img_BGR):

        block_number = 6  # 读取数字的个数

        img_BGR = cv2.medianBlur(img_BGR, 7)  # 中值滤波模糊化

        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)  # BGR改为HSV

        cutten_all = red_line_check(img_HSV, img_BGR)  # 红线检测和整体切割

        numbers = cut_tiny6(cutten_all)  # 切割六个小正方形线框

        post_card = []  # 保存数字
        two = []  # 保存二值化数字图片
        test_out = []  # 保存归一化图片矩阵

        for i in range(block_number):

            single_num = ad_cut_central(numbers[i])  # 单个数字居中处理

            out_image = reshape_std(single_num, 28, 28)  # 获取28*28数字矩阵
            out_image = (255-out_image).astype(int)  # 灰度反转

            out_image = out_image / 255  # 归一化

            test_out.append(out_image.reshape(28, 28, 1))

        images = np.stack(test_out)

        return images

    def predict(self, image, sess):

        sub_images = self.process_image(image)

        feed_dict = {
            self.net.x: sub_images
        }

        predictions = sess.run(self.net.pred, feed_dict)

        pred_num = np.argmax(predictions, axis=1)

        return pred_num


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    zip_obj = ZipId()

    with tf.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.getcwd())

        if ckpt and ckpt.model_checkpoint_path:
            # 如果保存过模型，则在保存的模型的基础上继续训练
            zip_obj.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Reload Successfully!')

        image = cv2.imread('./test/woc1.jpg')
        # 在这里传入图片
        pred = zip_obj.predict(image, sess)

        cv2.putText(image, str(pred), (0, int(image.shape[0]/4)),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow('a', image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
