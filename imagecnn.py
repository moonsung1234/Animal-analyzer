
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import numpy as np
import datetime
import math
import os

class ImageCnnModel :
    def __init__(self, learning_rate, loop_count, learning_number, inputs, outputs) :
        self.learning_rate = learning_rate
        self.loop_count = loop_count
        self.learning_number = learning_number

        tf.disable_eager_execution()

        self.input = inputs
        self.output = outputs
        self.x = tf.placeholder(tf.float32, [None, inputs], name="x")
        self.t = tf.placeholder(tf.float32, [None, outputs], name="t")

    def __setVariables(self) :
        reshape_wh = int(math.sqrt(self.input))

        #컨볼루션층 1
        self.a1 = tf.reshape(self.x, [-1, reshape_wh, reshape_wh, 1], name="a1")
        self.f1 = tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.01), name="f1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[20]), name="b1")
        self.c1 = tf.nn.conv2d(self.a1, self.f1, strides=[1, 1, 1, 1], padding="SAME", name="c1")
        self.z1 = tf.nn.relu(self.c1 + self.b1, name="z1")

        #컨볼루션층 2 
        self.a2 = tf.nn.max_pool(self.z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a2")
        self.f2 = tf.Variable(tf.random_normal([5, 5, 20, 40], stddev=0.01), name="f2")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[40]), name="b2")
        self.c2 = tf.nn.conv2d(self.a2, self.f2, strides=[1, 1, 1, 1], padding="SAME", name="c2")
        self.z2 = tf.nn.relu(self.c2 + self.b2, name="z2")

        #컨볼루션층 3
        self.a3 = tf.nn.max_pool(self.z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a3")
        self.f3 = tf.Variable(tf.random_normal([5, 5, 40, 60], stddev=0.01), name="f3")
        self.b3 = tf.Variable(tf.constant(0.1, shape=[60]), name="b3")
        self.c3 = tf.nn.conv2d(self.a3, self.f3, strides=[1, 1, 1, 1], padding="SAME", name="c3")
        self.z3 = tf.nn.relu(self.c3 + self.b3, name="z3")

        self.a4 = tf.nn.max_pool(self.z3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a4")

        #완전연결층
        self.a4_flat = tf.reshape(self.a4, [-1, 60 * int(reshape_wh / 8)**2], name="a4_flat")

        #출력층
        self.w = tf.Variable(tf.random_normal([60 * int(reshape_wh / 8)**2, self.output], stddev=0.01), name="w")
        self.b = tf.Variable(tf.random_normal([self.output]), name="b")

        self.z_out = tf.matmul(self.a4_flat, self.w) + self.b
        self.y = tf.nn.softmax(self.z_out)

    def __setOptimizer(self) :
        #set optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.z_out, labels=self.t))
        self.optimi = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimi.minimize(self.loss)

    def __setAccuracy(self) :
        #predict value and get accuracy
        self.predict_value = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.predict_value, dtype=tf.float32))

    def learn(self, x_all, t_all) :
        self.__setVariables()
        self.__setOptimizer()
        self.__setAccuracy()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.new_saver = tf.train.Saver()

        if os.path.exists("imagecnn.meta") :
            self.saver = tf.train.import_meta_graph("imagecnn.meta")
            self.saver.restore(self.sess, "imagecnn")
            #self.graph = tf.get_default_graph()

            """
            self.f1 = self.graph.get_tensor_by_name("f1:0")
            self.f2 = self.graph.get_tensor_by_name("f2:0")
            self.f3 = self.graph.get_tensor_by_name("f3:0")

            self.w = self.graph.get_tensor_by_name("w:0")
            self.b = self.graph.get_tensor_by_name("b:0")
            """

        #run session
        x_train, x_test, t_train, t_test = train_test_split(x_all, t_all, test_size = 0.2, random_state=1234)
        
        x = self.x
        t = self.t
        
        start_time = datetime.datetime.now()

        train_x = x_train
        train_t = t_train

        test_x = x_test
        test_t = t_test

        for i in range(self.loop_count) :
            for ii in range(int(len(train_x) / self.learning_number)) :                   
                size = self.learning_number / len(train_x)
                batch_x, _, batch_t, _ = train_test_split(train_x, train_t, train_size=size, random_state=1234)

                loss_value, _ = self.sess.run([self.loss, self.train], feed_dict={x : batch_x, t : batch_t})
                
                if ii % self.learning_number == 0 :
                    print("step : ", i, ", loss : ", loss_value)

        end_time = datetime.datetime.now()
        accuracy_value = self.sess.run(self.accuracy, feed_dict={x : test_x, t : test_t})
        
        self.new_saver.save(self.sess, "imagecnn")

        print("time : ", end_time - start_time)
        print("accuracy : ", accuracy_value)

    def predict(self, x_data) :
        self.__setVariables()
        self.__setOptimizer()
        self.__setAccuracy()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if os.path.exists("imagecnn.meta") :
            self.saver = tf.train.import_meta_graph("imagecnn.meta")
            self.saver.restore(self.sess, "imagecnn")
            #self.graph = tf.get_default_graph()

            """
            self.f1 = self.graph.get_tensor_by_name("f1:0")
            self.f2 = self.graph.get_tensor_by_name("f2:0")
            self.f3 = self.graph.get_tensor_by_name("f3:0")

            self.w = self.graph.get_tensor_by_name("w:0")
            self.b = self.graph.get_tensor_by_name("b:0")
            """

        x, t = self.x, self.t
        predict_value = self.sess.run(self.y, feed_dict={x : x_data})

        return predict_value
