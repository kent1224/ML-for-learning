# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:27:01 2017

@author: 14224
"""

""" ML x Tensorflow - template """

numbers_of_points = 500

""" show the true value of x and y """
a = 0.22
b = 0.78

x_points = []
y_points = []

for i in range(numbers_of_points):
    x = np.random.normal(0.0,0.5)
    
    y = a * x + b + np.random.normal(0.0,0.1)
    x_points.append([x])
    y_points.append([y])

plt.plot(x_points, y_points, 'o', label = 'input data')
plt.legend()
plt.show()

""" Now, estimate the model """
A = tf.Variable(tf.random_uniform([1],-1.0,1.0))
B = tf.Variable(tf.zeros([1]))

y = tf.add(tf.multiply(A,x_points),B)





import numpy as np
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt

""" Define parameters """
#群集所有的點數為1000
num_vectors = 1000
#分區的數量
num_clusters = 4
#演算法的計算步階
num_steps = 100
#講解裡沒有的
n_smaples_per_cluster = 500

""" Data """
#load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#split data into x, y, train, validation, test, ...
# 訓練資料用100個mnist影像 (train x, y)
train_pixels, train_list_value = mnist.train_next_batch(100)
# 測試資料用10個mnist影像 (test x, y)
test_pixels, test_list_value = mnist.test_next_batch(10)

""" 定義張量 """
train_pixels_tensor = tf.placeholder("float", [None,784])
test_pixels_tensor = tf.placeholder("float", [784])

""" Define the cost function and optimize it """
# cost function: distance, mean square error,  
distance = tf.reduce_sum(tf.abs(tf.add(train_pixels_tensor, tf.neg(test_pixels_tensor))),reduction_indices=1)
cost_function = tf.reduce_mean(tf.square(y-y_points))

# optimizer: Gradient Descent(learning_rate)
#arg_min回傳最小distance的索引
pred = tf.arg_min(distance, 0)
#Gradient Descnet
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)


""" Define evaluation function """
accuracy = 0

""" initialize variables """
init = tf.global_variables_initializer()

""" session: 測試與演算法評估 """
with tf.Session() as sess:  #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    # run initializer    
    sess.run(init)
    
    for i in range(len(test_list_value)):
        nn_index = sess.run(pred, feed_dict={train_pixels_tensor: train_pixels,\
                                             test_pixels_tensor: test_pixels[i,:]})
        print("Test N ", i, "predicted class: ", np.argmax(train_list_value(nn_index)), "true class: ", np.argmax(test_list_value[i]))
        
        if np.argmax(train_list_value(nn_index)) == np.argmax(test_list_value[i]):
            accuracy += 1./len(test_pixels)
    
    # print and plot results
    print("Result: ", accuracy)
    
    
        for step in range(0,21):
        sess.run(optimizer)
        
        if (step % 5) ==0:
            plt.plot(x_points, y_points, 'o', label = 'step = {}'.format(step))
            plt.plot(x_points, sess.run(A) * x_points+sess.run(B))
            plt.legend()
            plt.show()
            
            
            
            
            


""" 最後視覺化函數的內容 """
def display_partition(x_values, y_values, assignment_values):
    label = []
    colors = ["red", "blue", "green", "yellow"]
    for i in xrange(len(assignment_values)):
        label.append(colors[(assignment_values[i])])
    
    color = labels
    df = pd.DataFrame(dict(x = x_values, y = y_values, color = labels))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c = df['color'])
    plt.show()


""" Data """
#initialize the data
x_values = []
y_values = []
vector_values = []

#create random data
#numpy 的 random.normal允許我們建立x_values和y_values向量
for i in xrange(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4,0.7))
        y_values.append(np.random.normal(0.2,0.8))
    else:
        x_values.append(np.random.normal(0.6,0.4))
        y_values.append(np.random.normal(0.8,0.5))

#使用zip函數來獲得完整串列
vector_values = zip(x_values, y_values)

#透過tensorflow將vector_values轉換為一個常數
vectors = tf.constant(vector_values)

#視覺化
plt.plot(x_values, y_values, 'o', label = 'Input Data')
plt.legend()
plt.show()

""" 產生中心點這邊不懂 """
#產生4個中心點(k=4)，接著透過tf.random_shuffle決定索引
n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))
#決定4個隨機的索引
begin = [0,]
size = [num_clusters,]
size[0] = num_clusters
#擁有所有初始中心點的索引
centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))
"""                   """

""" Define cost function and optimize it """
#不懂   為了管理之前定義的張量、向量與中心點，我們使用了tensorflow的expand_dims函數，它能夠擴展兩個引數的大小
#這個函數允許你標準化兩個張量的形狀
expand_vectors = tf.expand_dims(vectors, 0)
expand_centroids = tf.expand_dims(centroids, 1)

#cost function: Euclidean distance
vectors_subtration = tf.sub(expand_vectors, expand_centroids)
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration, 2))
#跨張量的euclidean_distance的最短距離的索引值
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

#講解裡沒有的
partitions = [0, 0, 1, 1, 0]
num_partitions = 2
data = [10, 20, 30, 40, 50]
outputs[0] = [10, 20, 50]
outputs[1] = [30, 40]

#使用assignments索引來分割向量到num_clusters張量
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
#更新中心點(使用tf.concat去串接成為一個張量)
update_centroids = tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions])

""" 測試與演算法評估 """
#initialize all variables
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

#開始計算
for step in xrange(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

#顯示結果
display_partition(x_values, y_values, assignment_values)
plt.plot(x_values, y_values, 'o', label = 'Input Data')
plt.legend()
plt.show()