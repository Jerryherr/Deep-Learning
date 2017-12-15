
# coding: utf-8

# In[36]:

import cv2 as cv
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time


# In[37]:

data_path = 'E:\\6000B - Deep Learning\\project3\\Dataset_A\\'
file_path = 'E:\\6000B - Deep Learning\\project3\\Dataset_A\\'
model_path = 'E:\\6000B - Deep Learning\\project3\\model\\'
color = 0


# In[38]:

# 图片和标签的字典 {img ： label}
img_label={}
with open(file_path + 'train.txt') as f:
    for line in f:
        line = line.strip('\n')
        img = line.split('\t')[0]
        label = line.split('\t')[1]
        img_label[img] = float(label)
        
img_label_val={}
with open(file_path + 'val.txt') as f:
    for line in f:
        line = line.strip('\n')
        img = line.split('\t')[0]
        label = line.split('\t')[1]
        img_label_val[img] = float(label)
        
img_label_test={}
with open(file_path + 'test.txt') as f:
    for line in f:
        line = line.strip('\n')
        img_label_test[line] = 0


# In[39]:

sample_img = []
for name in list(img_label.keys())[:]:
#for name in ['22614074','22614097','22614127','22614150']:
    for img in os.listdir(data_path):
        if img.startswith(name):
            sample_img.append([name,cv.imread(data_path+img,color)])
print('finish loading training data')

sample_img_val = []
for name in list(img_label_val.keys())[:]:
#for name in ['24055024','24054997','24065270','20587080']:
    for img in os.listdir(data_path):
        if img.startswith(name):
            sample_img_val.append([name,cv.imread(data_path+img,color)])
print('finish loading validation data')
            
sample_img_test = []
for name in list(img_label_test.keys())[:]:
#for name in ['24055024','24054997','24065270','20587080']:
    for img in os.listdir(data_path):
        if img.startswith(name):
            sample_img_test.append([name,cv.imread(data_path+img,color)])
print('finish loading testing data')


# In[40]:

# 垂直对乳房轮廓画条切线，把不包括乳房的部分去除
filtered_img = []
for img in sample_img:
    col_2_keep = []
    col_index = 0
    for img_col in img[1].T:
        if (img_col>0).any():
            col_2_keep.append(col_index)
        col_index += 1
    filtered_img.append([img[0],img[1].T[col_2_keep].T])
print('finish filtering training data')

    
filtered_img_val = []
for img in sample_img_val:
    col_2_keep = []
    col_index = 0
    for img_col in img[1].T:
        if (img_col>0).any():
            col_2_keep.append(col_index)
        col_index += 1
    filtered_img_val.append([img[0],img[1].T[col_2_keep].T])
print('finish filtering validation data')

filtered_img_test = []
for img in sample_img_test:
    col_2_keep = []
    col_index = 0
    for img_col in img[1].T:
        if (img_col>0).any():
            col_2_keep.append(col_index)
        col_index += 1
    filtered_img_test.append([img[0],img[1].T[col_2_keep].T])
print('finish filtering testing data')


# In[41]:

# 分patch
patches = []
patch_w = 250
patch_h = 250
for img in filtered_img:
    
    # 图片（丢掉黑色部分）的长宽
    filtered_img_h = img[1].shape[0]
    filtered_img_w = img[1].shape[1]
    
    patch_top_left_corner_y = 0
    # 一行一行往下扫
    while (patch_top_left_corner_y + patch_h) < filtered_img_h:
        
        # 一列一列往右扫
        patch_top_left_corner_x = 0
        while (patch_top_left_corner_x + patch_w) < filtered_img_w:
            patch = img[1][patch_top_left_corner_y : patch_top_left_corner_y + patch_h,
                           patch_top_left_corner_x : patch_top_left_corner_x + patch_w ]
            
            if (patch>0).all():
                # [bag, label, 坐标[左上角行数，左上角列数]，patch的每个像素]
                # normalize patch
                #patch = (patch - patch.min())/ (patch.max() -  patch.min())
                patches.append([img[0],img_label[img[0]],patch.reshape((patch_h,patch_w,1))])
            
            patch_top_left_corner_x += patch_w
        patch_top_left_corner_y += patch_h

print('finish retrieving patches of training data')
patches_val = []
for img in filtered_img_val:
    
    # 图片（丢掉黑色部分）的长宽
    filtered_img_h = img[1].shape[0]
    filtered_img_w = img[1].shape[1]
    
    patch_top_left_corner_y = 0
    # 一行一行往下扫
    while (patch_top_left_corner_y + patch_h) < filtered_img_h:
        
        # 一列一列往右扫
        patch_top_left_corner_x = 0
        while (patch_top_left_corner_x + patch_w) < filtered_img_w:
            patch = img[1][patch_top_left_corner_y : patch_top_left_corner_y + patch_h,
                           patch_top_left_corner_x : patch_top_left_corner_x + patch_w ]
            
            if (patch>0).all():
                # [bag, label, 坐标[左上角行数，左上角列数]，patch的每个像素]
                #patch = (patch - patch.min())/ (patch.max() -  patch.min())
                patches_val.append([img[0],img_label[img[0]],patch.reshape((patch_h,patch_w,1))])
            
            patch_top_left_corner_x += patch_w
        patch_top_left_corner_y += patch_h

print('finish retrieving patches of validation data')
patches_test = []
for img in filtered_img_test:
    
    # 图片（丢掉黑色部分）的长宽
    filtered_img_h = img[1].shape[0]
    filtered_img_w = img[1].shape[1]
    
    patch_top_left_corner_y = 0
    # 一行一行往下扫
    while (patch_top_left_corner_y + patch_h) < filtered_img_h:
        
        # 一列一列往右扫
        patch_top_left_corner_x = 0
        while (patch_top_left_corner_x + patch_w) < filtered_img_w:
            patch = img[1][patch_top_left_corner_y : patch_top_left_corner_y + patch_h,
                           patch_top_left_corner_x : patch_top_left_corner_x + patch_w ]
            
            if (patch>0).all():
                # [bag, label, 坐标[左上角行数，左上角列数]，patch的每个像素]
                #patch = (patch - patch.min())/ (patch.max() -  patch.min())
                patches_test.append([img[0],img_label[img[0]],patch.reshape((patch_h,patch_w,1))])
            
            patch_top_left_corner_x += patch_w
        patch_top_left_corner_y += patch_h
        
print('finish retrieving patches of testing data')


# In[42]:

len(patches_val)


# In[43]:

len(patches_test)


# In[44]:

len(patches)


# In[45]:

patches_img = []
patches_y = []
patches_x = []

for p in patches:
    patches_img.append(p[0])
    patches_y.append(p[1])
    patches_x.append(p[2])

patches_img = np.array(patches_img).reshape((-1,1))
patches_y = np.array(patches_y).reshape((-1,1))
patches_x = np.array(patches_x)

patches_img_val = []
patches_y_val = []
patches_x_val = []

for p in patches_val:
    patches_img_val.append(p[0])
    patches_y_val.append(p[1])
    patches_x_val.append(p[2])

patches_img_val = np.array(patches_img_val).reshape((-1,1))
patches_y_val = np.array(patches_y_val).reshape((-1,1))
patches_x_val = np.array(patches_x_val)

patches_img_test = []
patches_y_test = []
patches_x_test = []

for p in patches_val:
    patches_img_test.append(p[0])
    patches_y_test.append(p[1])
    patches_x_test.append(p[2])

patches_img_test = np.array(patches_img_test).reshape((-1,1))
patches_y_test = np.array(patches_y_test).reshape((-1,1))
patches_x_test = np.array(patches_x_test)


# In[46]:

def minibatch(train_img,train_label,train_name,num_data,batch_size):
    batches = []
    num_of_minibatch = int(num_data/batch_size)
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    for i in range(num_of_minibatch):
        ind = indices[(i*batch_size):((i+1)*batch_size)]
        feature_batch = train_img[ind,:,:,:]
        label_batch = train_label[ind,:]
        name_batch = train_name[ind,:]
        batches.append([feature_batch,label_batch,name_batch])
        
    return batches   


# In[61]:

batch_size = 256
importance = 0.6
filter_w  = 10
filter_h  = 10
feature_map = 3
fc_node = 128
l_r = 0.05
epoch = 3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.6)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

##################################################
# 选重要的patch
##################################################
x_image = tf.placeholder(tf.float32, [None,patch_w,patch_h,1])

# 卷积1
W_conv1_1 = weight_variable([filter_w, filter_h, 1, feature_map])
b_conv1_1 = bias_variable([feature_map])
h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
print('h_conv1_1: ',str(h_conv1_1.shape))

# 全连接1
dim = int(h_conv1_1.shape[1] * h_conv1_1.shape[2] * h_conv1_1.shape[3])
W_fc1_1 = weight_variable([dim,1])
b_fc1_1 = bias_variable([1])

h_conv1_1_flat = tf.reshape(h_conv1_1, [-1, dim])
print('h_conv1_1_flat: ',str(h_conv1_1_flat.shape))
y_conv_1_1 = tf.nn.sigmoid(tf.matmul(h_conv1_1_flat, W_fc1_1) + b_fc1_1)
print('y_conv_1_1: ',str(y_conv_1_1.shape))

values, indices = tf.nn.top_k(tf.reshape(y_conv_1_1,(1,-1)), k=round(batch_size*importance), sorted=True)
indices = tf.reshape(indices,[-1])
selected_x_image = tf.gather(x_image,indices)



# # 对着label的0,1做变换，然后挑出前60%
# y_ = tf.placeholder(tf.float32, [batch_size, 1])
# y_conv_1_1 = tf.multiply(y_conv_1_1,y_) + (1 - y_)*(1-y_conv_1_1)
# print('y_conv_1_1: ',str(y_conv_1_1.shape))

# values, indices = tf.nn.top_k(tf.reshape(y_conv_1_1,(1,-1)), k=round(batch_size*importance), sorted=True)
# #nn_output = round(batch_size*importance) - tf.reduce_sum(values)

# indices = tf.reshape(indices,[-1])
# print(indices.shape)
# selected_x_image = tf.gather(x_image,indices)
# print(selected_x_image.shape)

##################################################
# 挑出了头60%个重要的,然后把这些重要的再扔进另一个cnn里做预测
##################################################


# 卷积1
W_conv2_1 = weight_variable([filter_w, filter_h, 1, feature_map])
b_conv2_1 = bias_variable([feature_map])
h_conv2_1 = tf.nn.relu(conv2d(selected_x_image, W_conv2_1) + b_conv2_1)
print('h_conv2.1: ',str(h_conv2_1.shape))

h_pool2_1 = max_pool_2x2(h_conv2_1)
print('h_pool2_1: ',str(h_pool2_1.shape))

# 卷积2
W_conv2_2 = weight_variable([filter_w, filter_h, feature_map, feature_map])
b_conv2_2 = bias_variable([feature_map])
h_conv2_2 = tf.nn.relu(conv2d(h_pool2_1, W_conv2_2) + b_conv2_2)
print('h_conv2_2: ',str(h_conv2_2.shape))

h_pool2_2 = max_pool_2x2(h_conv2_2)
print('h_pool2_2: ',str(h_pool2_2.shape))

# 全连接1
dim = int(h_pool2_2.shape[1] * h_pool2_2.shape[2] * h_pool2_2.shape[3])
W_fc2_1 = weight_variable([dim,fc_node])
b_fc2_1 = bias_variable([fc_node])

h_pool2_2_flat = tf.reshape(h_pool2_2, [-1, dim])
print('h_pool2_2_flat: ',str(h_pool2_2_flat.shape))
y_conv_2_1 = tf.nn.relu(tf.matmul(h_pool2_2_flat, W_fc2_1) + b_fc2_1)
print('y_conv_2_1: ',str(y_conv_2_1.shape))

# 全连接2
W_fc2_2 = weight_variable([fc_node,1])
b_fc2_2 = bias_variable([1])

reduce = tf.reshape(tf.matmul(y_conv_2_1, W_fc2_2) + b_fc2_2,[-1])
y_conv_2_2 = tf.nn.softmax(reduce)
print('y_conv_2_2: ',str(y_conv_2_2.shape))

# 标签和被选中的标签
y_ = tf.placeholder(tf.float32, [None, 1])
y = tf.reshape(y_,[-1])
selected_y_ = tf.gather(y,indices)

# 图片的名字和被选中的图片名字
img_name_ = tf.placeholder(tf.float32, [None, 1])
img_name = tf.reshape(img_name_,[-1])
selected_img_name = tf.gather(img_name,indices)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = selected_y_ ,logits= y_conv_2_2))
#cross_entropy = tf.reduce_mean(tf.pow((selected_y_ - y_conv_2_2),2))
train_step = tf.train.AdamOptimizer(l_r).minimize(cross_entropy)

#correct_prediction = tf.gather(y_,indices) - y_conv_2_2
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[62]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
start_time = time.time()

for e in range(epoch):
    train_minibatches = minibatch(patches_x,patches_y,patches_img,len(patches),batch_size)
    val_minibatches = minibatch(patches_x_val,patches_y_val,patches_img_val,len(patches_val),batch_size)
    step = 0    
    for b in train_minibatches:
        step += 1
        train_step.run(feed_dict ={x_image: b[0], y_: b[1]})
        if step % 1 == 0 and step>0:
            print('epoch: ' , str(e), ' step: ' , str(step))
            print('training loss:')
            print(sess.run(cross_entropy,feed_dict = {x_image: b[0], y_: b[1], img_name_: b[2]}))
            print('validation loss:')
            loss_val = 0
            for b_val in val_minibatches:
                loss_val += sess.run(cross_entropy,feed_dict = {x_image: b_val[0], y_: b_val[1], img_name_: b_val[2]})
            print(loss_val/(len(patches_val)/batch_size))
            print('prediction: ')
            print(sess.run(y_conv_2_2,feed_dict = {x_image: b[0], y_: b[1], img_name_: b[2]}))
            print(sess.run(y_conv_2_2,feed_dict = {x_image: b[0], y_: b[1], img_name_: b[2]}))
            print('total time elapsed: ' + str(time.time()-start_time))
            print('********************************************************')



# In[63]:

result,from_img = sess.run([y_conv_2_2,selected_img_name],feed_dict = {x_image: patches_x_test,img_name_: patches_img_test})
#from_img = sess.run(selected_img_name,feed_dict = {x_image: patches_x_test})

# save model
save_path = saver.save(sess, model_path + 'model.ckpt')
print("Model saved in file: %s" % save_path)


# In[ ]:

#write to csv
prediction = []
img_name = []
cnt = 0
for r in result:
    img_name.append(from_img[cnt,0])
    p = sum(list(r))/len(list(r))
    if p > 0.5:
        prediction.append(p)
    cnt += 1
    
with open(model_path + 'project2_20451451.csv','w', newline='') as f:
    writer = csv.writer(f)
    for k,v in enumerate(predition):
        writer.writerows([img_name[k],str(v)])


# In[ ]:



