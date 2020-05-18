# Here, I do NLP with LSTM, CNN and Dense, with L1 Regularization on Dense, L2 Regularization on all Nets and BatchNormalization.

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential
batchsz = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
print(x_train.shape)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

net_embedding = Sequential([layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len),
                     
                            layers.Dropout(0.5), 
                     
                            ])

net_conv = Sequential([ layers.Conv2D(126, (3, embedding_len), strides=1, padding='valid')
                     
                        ])

net_batch= Sequential([ layers.Dropout(0.5), 
                     
                        layers.BatchNormalization()])

net_LSTM = Sequential([layers.LSTM(64, dropout=0.5, return_sequences=True, unroll=True),
            
                       layers.LSTM(64, dropout=0.5, unroll=True)
                     
                        ])
net_Dense=Sequential([layers.Dense(1)])
optimizer = optimizers.Adam(lr=1e-3)

net_embedding.build(input_shape=[None, 80])
net_conv.build(input_shape=[None, 80,100,1])
net_batch.build(input_shape=[None, 78,1,126])
net_LSTM.build(input_shape=[None, 78,126])
net_Dense.build(input_shape=[None, 64])

variables_Dense_Conv = net_Dense.trainable_variables + net_conv.trainable_variables
variables = net_embedding.trainable_variables + net_conv.trainable_variables + net_LSTM.trainable_variables + net_Dense.trainable_variables + net_batch.trainable_variables

for epoch in range(100):
     for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            y=tf.cast(y,dtype=tf.float32)
            y=tf.expand_dims(y,axis=-1)
            
            out1=net_embedding(x)
            
            out1=tf.expand_dims(out1,axis=-1)
            
            out2=net_conv(out1)
            out2=net_batch(out2,training=True)
            out2=tf.squeeze(out2,axis=2)
            
            out3=net_LSTM(out2)
            #print(out3.shape)
            out3=net_Dense(out3)
                        
            loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=out3)
            
            #l1 regularization for weights in net_Dense
            r=[]
            for p in range(0,len(net_Dense.trainable_variables),2):
                r.append(tf.norm(net_Dense.trainable_variables[p],ord=1))
               
            r=tf.reduce_sum(tf.stack(r))
            
            #l2 regularization for weights in net_conv and net_Dense
            s=[]
            for q in range(0,len(variables_Dense_Conv),2):        
                s.append(tf.nn.l2_loss(variables_Dense_Conv[q]))
            s=tf.reduce_sum(tf.stack(s))
            
            #l2 regularization for weights in net_LSTM
            t=[]
            for i in range(len(net_LSTM.trainable_variables)):        
                t.append(tf.nn.l2_loss(net_LSTM.trainable_variables[i]))
            t=tf.reduce_sum(tf.stack(s))
            
            loss = tf.reduce_mean(loss)
            loss=loss+0.001*r+0.001*s+0.001*t

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step %100 == 0:
            print(epoch, step, 'loss:', float(loss))
    
     total_num = 0
     total_correct = 0
     for x,y in db_test:
            y=tf.cast(y,dtype=tf.int32)
            y=tf.expand_dims(y,axis=-1)
            out1=net_embedding(x)
            
            out1=tf.expand_dims(out1,axis=-1)
            
            out2=net_conv(out1)
            out2=net_batch(out2,training=False)
            out2=tf.squeeze(out2,axis=2)
            
            out3=net_LSTM(out2)
            out3=net_Dense(out3)
            
            out3=tf.nn.sigmoid(out3)
            #round numbers to 0 or 1
            out3=tf.round(out3)
            
            out3 = tf.cast(out3, dtype=tf.int32)

            correct = tf.cast(tf.equal(out3, y), dtype=tf.int32)
            
            correct = tf.reduce_sum(correct)
            
            total_num += x.shape[0]
            total_correct += int(correct)
     

     acc = total_correct / total_num
     print(epoch, 'acc:', acc)
