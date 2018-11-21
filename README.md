import numpy as np
import tensorflow as tf
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存

color = 100/255


def get_imagelist(path):   # 此函数读取特定文件夹下的bmp格式图像

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


def judgeedge(img, dis, flag, size):
    for i in range(dis):
        # Cow or Column 判断是行是列
        if flag == 0:
            line1 = img[i, :] < color
            line2 = img[dis-1-i, :] < color
        else:
            line1 = img[:, i] < color
            line2 = img[:, dis-1-i] < color
        # If edge, recode serial number 若有手写数字，即到达边界，记录下行
        if sum(line1) >= 1 and size[0] == -1:
            size[0] = i
        if sum(line2) >= 1 and size[1] == -1:
            size[1] = dis-1-i
        # If get the both of edge, break 若上下边界都得到，则跳出
        if size[0] != -1 and size[1] != -1:
            break
    return size


# Cut the Picture 切割图象
def cutpicture(img):
    size = []
    # 图片的行数
    length = len(img)
    # 图片的列数
    width = len(img[0, :])
    # 计算新大小
    size.append(judgeedge(img, length, 0, [-1, -1]))
    size.append(judgeedge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    print(size)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]


imagelist = get_imagelist(r"E:\sym1")
imagedata = []
imageshape0 = []
imageshape1 = []

for k in range(len(imagelist)):
    imagedata.append(np.array(Image.open(imagelist[k]).resize((128, 128)).convert('L')).astype('float')/255)
    imagedata[k] = cutpicture(imagedata[k])
    imageshape0.append(imagedata[k].shape[0])
    imageshape1.append(imagedata[k].shape[1])

imageshape = imageshape0+imageshape1

mul = max(imageshape)
while mul % 4 != 0:
    mul = mul+1


def stretchpicture(img):
    newimg = np.ones(mul**2).reshape(mul, mul)
    # The length of each cows after stretching 每一行拉伸/压缩的步长
    step1 = (len(img[0])-1)/mul
    # columns
    step2 = (len(img)-1)/mul
    # Operate on each cows 对每一行进行操作
    for i in range(len(img)):
        for j in range(mul-1):
            newimg[i, j] = img[i, int(np.floor(j*step1))]
    # Operate on each columns 对每一列进行操作
    for i in range(len(img[0])):
        for j in range(mul-1):
            newimg[j, i] = img[int(np.floor(j*step2)), i]
    return newimg


for k in range(len(imagedata)):
    imagedata[k] = stretchpicture(imagedata[k])

Q = np.loadtxt(r'E:\sym1\symQ.csv', delimiter=',')  # label
syms = imagedata[1].shape[1]
cats = Q.shape[1]
x = tf.placeholder('float', [None, syms, syms])
y = tf.placeholder('float', [None, cats])


# train
# 初始化权重和bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(xin, win):
    return tf.nn.conv2d(xin, win, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(xin):
    return tf.nn.max_pool(xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 转化为正方形图片
x_re = tf.reshape(x, [-1, int(mul), int(mul), 1])

# 第一层
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_re, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 第二层
W_conv2 = weight_variable([6, 6, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_pool2_flat = tf.reshape(h_pool2, [-1, int(mul*mul*4)])

# 第三层
W_conv3 = weight_variable([12, 12, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_flat = tf.reshape(h_pool3, [-1, int(mul*mul*2)])

# 全连接层
W_fc1 = weight_variable([int(mul*mul*2), 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, cats])
b_fc2 = bias_variable([cats])
y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2
# 将权值转化为概率，此处方法为矩阵中所有元素减去矩阵中最小值，再除以矩阵的最大值减最小值之差
y_conv2 = tf.transpose((tf.transpose(y_conv)-tf.reduce_min(y_conv))/(tf.reduce_max(y_conv)-tf.reduce_min(y_conv)))
# 转化为0-1表达的Q矩阵
y_conv1 = tf.round(y_conv2)

# 损失函数，此处是预测结果与实际需求的交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv2))

# 梯度下降优化
train_step_1 = tf.train.AdadeltaOptimizer(learning_rate=tf.train.exponential_decay(1.0, tf.Variable(tf.constant(0)), 20,
                                                                                   0.96, staircase=True)).minimize(loss)

# 计算精度，标准为转化为0-1的预测结果和实际Q矩阵
correct_prediction = tf.equal(y, y_conv1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存模型
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()

# 初始化参数
init = tf.global_variables_initializer()


with tf.Session(config=config) as sess:
    sess.run(init)
    accuracy_l = 0

    for epoch in range(0,1000):
        sess.run(train_step_1, feed_dict={x: imagedata, y: Q, keep_prob: 0.5})
        cm = sess.run(y_conv, feed_dict={x: imagedata, y: Q, keep_prob: 1.0})
        accuracy_n = sess.run(accuracy, feed_dict={x: imagedata, y: Q, keep_prob: 1.0})
        print("第" + str(epoch+1) + "轮，准确率为：" + str(accuracy_n))

        # 两次改变小于某阈值且总体正确率高于0.8时，终止程序并保存学习内容。
        if abs(accuracy_l - accuracy_n) < 0.0001 and accuracy_n > 0.98:
            # 保存训练模型
            global_step.assign(epoch).eval()
            saver.save(sess, "D:/testmodelseve/model.ckpt", global_step=global_step)
            nit = str(epoch)
            # print(sess.run(y_conv, feed_dict={x: imagedata, y: Q, keep_prob: 1.0}))
            # print(np.round(np.transpose((np.transpose(cm)-cm.min())/(cm.max()-cm.min()))))
            # print(sess.run(tf.argmax(y_conv, 1), feed_dict={x: imagedata, y: Q, keep_prob: 1.0}))
            print("work complete")
            break
        if accuracy_n == 0:
            print("mission failed, please restart")
            break
        # if accuracy_l <= accuracy_n:
        accuracy_l = accuracy_n
        if epoch == 999:
            global_step.assign(epoch).eval()
            saver.save(sess, "D:/testmodelseve/model.ckpt", global_step=global_step)
            nit = str(epoch)
            # print(sess.run(y_conv, feed_dict={x: imagedata, y: Q, keep_prob: 1.0}))
            # print(np.round(np.transpose((np.transpose(cm) - cm.min()) / (cm.max() - cm.min()))))
            # print(sess.run(tf.argmax(y_conv, 1), feed_dict={x: imagedata, y: Q, keep_prob: 1.0}))
            print("better than nothing")

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)


# 使用保存的训练结果进行预测
with tf.Session() as sess:
    sess.run(init)
    # noinspection PyUnboundLocalVariable
    saver.restore(sess, 'D:/testmodelseve/model.ckpt-'+nit)
    y_predict = y_conv.eval(feed_dict={x: imagedata, keep_prob: 1.0})
    predict_out = np.round(np.transpose((np.transpose(y_predict) - y_predict.min()) / (
            y_predict.max() - y_predict.min())))
