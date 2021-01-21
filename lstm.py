#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.preprocessing import StandardScaler
tf.compat.v1.disable_eager_execution()
#创建数据
f=open('daydata.csv')
df=pd.read_csv(f)     
data=np.array(df['xiaomai'])
# print(data)
#data=data[::-1]
data=data[0:]
plt.figure()
plt.plot(data)
plt.show()
#标准化
normalize_data=(data-np.mean(data))/np.std(data)#标准化
normalize_data=normalize_data[:,np.newaxis]#增加维度
#参数，步长
iterations=1000  #We can increase the number of iterations to gain better result.
time_step=10#2时间步
rnn_unit=10      #hiddenlayer
lstm_layers=2
batch_size=20 #20
input_size=1      
output_size=1     
lr=0.0006         
train_x,train_y=[],[]
#前多少数据来预测后面数据
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 

#数据转换成【1，2，3】
X=tf.compat.v1.placeholder(tf.float32, [None,time_step,input_size])
Y=tf.compat.v1.placeholder(tf.float32, [None,time_step,output_size])

weights={
         'in':tf.Variable(tf.random.normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random.normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


def lstm(batch):      
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_unit) for i in range(lstm_layers)])
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.compat.v1.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def train_lstm():
    global batch_size
    with tf.compat.v1.variable_scope("sec_lstm"):
        pred,_=lstm(batch_size)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        Loss=[]
        for i in range(iterations): #We can increase the number of iterations to gain better result.
            step=0
            start=0
            end=start+batch_size
            loss_temp=[]
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                
                if step%10==0:
                    print("Number of iterations:",i," loss:",loss_)
                    loss_temp.append(loss_)
                    #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    #if you run it in Linux,please use  'model_save1/modle.ckpt'
                step+=1
            Loss.append(np.mean(loss_))

        print("model_save", saver.save(sess, 'model_save1\\modle.ckpt'))
        print("The train has finished")
        return Loss,pred

Loss,pred=train_lstm()


def prediction():
    with tf.compat.v1.variable_scope("sec_lstm",reuse=tf.compat.v1.AUTO_REUSE):
        pred,_=lstm(1)    
    saver=tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, 'model_save1\\modle.ckpt') 
        #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        #if you run it in Linux,please use  'model_save1/modle.ckpt'
        prev_seq=train_x[0]

        predict=[]
        Predict=[]
        #基于训练集的预测
        for i in range(1,len(train_x)):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            temp=next_seq[-1]*np.std(data)+np.mean(data)
            Predict.append(temp[0])
            predict.append(next_seq[-1])
            # prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
            prev_seq = train_x[i]

        #预测后100天的
        prev_seq=train_x[-1]
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            temp = next_seq[-1] * np.std(data) + np.mean(data)
            Predict.append(temp[0])
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        #标准化数据还原，绘图
        plt.figure()
        plt.plot(list(range(len(normalize_data))), data, color='b',label="Original")
        plt.plot(list(range(len(Predict))),Predict, color='r',label='Predict(+100 days)')
        plt.legend()

        plt.savefig("预测图.jpg")
        plt.show()

    return Predict



        
predict=prediction()
res=pd.DataFrame({
    "yucezhi":predict
})

res["zhenshizhi"]= pd.DataFrame(data)
res.to_csv("ymjgycz.csv",index=None)

Loss=pd.DataFrame({
    "迭代次数":[i for i in range(iterations)],
    "Loss":Loss
})
Loss.to_csv("Loss.csv",index=None)
plt.plot(Loss.迭代次数,Loss.Loss)
plt.show()

# # print(len(pre))
# mse = mean_squared_error(data,pre )
# # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
# rmse = math.sqrt(mean_squared_error(data,pre))
# # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
# mae = mean_absolute_error(data, pre)
# print('均方误差: %.6f' % mse)
# print('均方根误差: %.6f' % rmse)
# print('平均绝对误差: %.6f' % mae)


