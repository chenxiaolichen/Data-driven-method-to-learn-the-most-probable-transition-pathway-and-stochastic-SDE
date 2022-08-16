#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
import time
import matplotlib.pyplot as plt
#import pickle5 as pickle


# In[ ]:


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1  
    H=X
#    H = 2.0*(X -  X.min(0))/( X.max(0) -  X.min(0)) - 1.0
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y =  tf.add(tf.matmul(H, W), b)
    return Y


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=0), dtype=tf.float64)


# In[ ]:


def neural_net_u(X, weights, biases):
    u= neural_net(X, weights, biases)
    return u

def neural_net_v(X, weights, biases):
    v= neural_net(X, weights, biases)
    return v


# In[ ]:


def net_f(tf1,weights_u,biases_u,weights_v,biases_v,sigma1,sigma2,lambda1,lambda2,lambda3,lambda4 ):
    u=neural_net(tf1, weights_u, biases_u)
    v=neural_net(tf1, weights_v, biases_v)
    u_t = tf.gradients(u, tf1)[0]
    u_tt = tf.gradients(u_t, tf1)[0]
    v_t = tf.gradients(v, tf1)[0]
    v_tt = tf.gradients(v_t, tf1)[0]
    #lambda1,lambda2,lambda3,lambda4,lambda5=1,-1,-1,-1,-1
    g1=lambda1*u+lambda2*u**3+lambda3*u*v**2
    g1_u=lambda1+3*lambda2*u**2+lambda3*v**2
    g1_v=2*lambda3*u*v
    g1_uu=6*lambda2*u
    g1_uv=2*lambda3*v

    g2=lambda4*v+lambda3*v*u**2
    g2_u=2*lambda3*v*u
    g2_v=lambda4+lambda3*u**2
    g2_vv=0
    g2_uv=2*lambda3*u
    

    return u_tt-v_t*(g1_v-sigma1**2/sigma2**2*g2_u)-g1*g1_u-sigma1**2/sigma2**2*g2*g2_u,v_tt-u_t*(g2_u-sigma2**2/sigma1**2*g1_v)-g2*g2_v-sigma2**2/sigma1**2*g1*g1_v


# In[ ]:


layers = [1] + 2* [20] + [1]
L = len(layers)
#tt1=time.time()
np.random.seed(0)

weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]    
biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]

weights_v = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]    
biases_v = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]


# In[ ]:


sigma1=tf.Variable([1],dtype=tf.float64,trainable=False)
sigma2=tf.Variable([1],dtype=tf.float64,trainable=False)
lambda1=tf.Variable([2],dtype=tf.float64,trainable=True)
lambda2=tf.Variable([-2],dtype=tf.float64,trainable=True)
lambda3=tf.Variable([-2],dtype=tf.float64,trainable=True)
lambda4=tf.Variable([-2],dtype=tf.float64,trainable=True) 
t0=0
t1=10
Nt=1000
vect=np.linspace(t0,t1,Nt+1)[:,None]
tf1_tf = tf.to_double(vect) 


# observation data

# In[ ]:


a=np.linspace(0,Nt,51,dtype=int)


# In[ ]:


t_ob=vect[a]
t_ob_tf = tf.to_double(t_ob) 


# In[ ]:


u_ob_nn=neural_net(t_ob_tf, weights_u, biases_u)
v_ob_nn=neural_net(t_ob_tf, weights_v, biases_v)
u_opt=np.loadtxt('/users/xchen104/2021/LearnSDE/2d-chemical/forward/result/u_opt-mat.txt');
v_opt=np.loadtxt('/users/xchen104/2021/LearnSDE/2d-chemical/forward/result/v_opt-mat.txt');
#u_opt=np.loadtxt('u_opt-mat.txt');
#v_opt=np.loadtxt('v_opt-mat.txt');
u_ob=u_opt[a][:,None];
v_ob=v_opt[a][:,None];


# In[ ]:


plt.plot(u_ob)


# In[ ]:


loss_u=tf.reduce_mean(tf.square(u_ob_nn-u_ob))
loss_v=tf.reduce_mean(tf.square(v_ob_nn-v_ob))
f_pred = net_f(tf1_tf,weights_u,biases_u,weights_v,biases_v,sigma1,sigma2,lambda1,lambda2,lambda3,lambda4 )


# In[ ]:


loss_ode1 = tf.reduce_mean(tf.square(f_pred[0]))
loss_ode2=tf.reduce_mean(tf.square(f_pred[1]))

loss=(loss_ode1 +loss_u)*1+loss_ode2 +loss_v


# In[ ]:
## choose a opt

optimizer_Adam = tf.train.AdamOptimizer(1e-4)
train_op_Adam = optimizer_Adam.minimize(loss)

loss_record = []
loss_ode1_record = []
loss_ode2_record = []
loss_u_record = []
loss_v_record = []
sigma1_record = []
sigma2_record = []
lambda1_record = []
lambda2_record = []
lambda3_record = []
lambda4_record = [] 
saver = tf.train.Saver(max_to_keep=1000)
savedir='xiaoli'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
min_loss = 1e16


# In[ ]:


with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    u_pred=neural_net(tf1_tf,weights_u,biases_u) 
    v_pred=neural_net(tf1_tf,weights_v,biases_v) 
    for i in range(200001):

        sess.run(train_op_Adam )
        if i % 100 == 0:
            (loss_result, loss_ode1_result, loss_u_result , loss_ode2_result, loss_v_result ) = sess.run([loss,
                    loss_ode1, loss_u,loss_ode2, loss_v])
            ut0= sess.run(u_pred)
            vt0= sess.run(v_pred)
            (temp_sigma1,temp_sigma2,temp_lambda1,temp_lambda2,temp_lambda3,temp_lambda4 )=sess.run([sigma1,sigma2,lambda1,lambda2,lambda3,lambda4 ])
            loss_record.append(loss_result)
            loss_ode1_record.append(loss_ode1_result)
            loss_u_record.append(loss_u_result)
            loss_ode2_record.append(loss_ode2_result)
            loss_v_record.append(loss_v_result)
            sigma1_record.append(temp_sigma1)
            sigma2_record.append(temp_sigma2)
            lambda1_record.append(temp_lambda1)
            lambda2_record.append(temp_lambda2)
            lambda3_record.append(temp_lambda3)
            lambda4_record.append(temp_lambda4) 
            if loss_result<min_loss:
                min_loss=loss_result
                u_opt= sess.run(u_pred) 
                v_opt= sess.run(v_pred)
                i_opt=i
           # temp_loss=sess.run(loss, feed_dict = all_dict)
            print ('  %d  %8.2e  %8.2e %8.2e  %8.2e  %8.2e %8.2e %8.2e  %8.2e %8.2e   %8.2e %8.2e ' % (i, loss_result,loss_ode1_result,loss_ode2_result, loss_u_result, loss_v_result,temp_sigma1,temp_sigma2,temp_lambda1,temp_lambda2,temp_lambda3,temp_lambda4 ) )

        if i % 50000 == 0:
            save_path = saver.save(sess, savedir+'/' + str(i) + '.ckpt')
            (weights_u_np,biases_u_np,weights_v_np,biases_v_np )=sess.run([weights_u,biases_u,weights_v,biases_v ])
            sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np,"weights_v": weights_v_np, "biases_v": biases_v_np}
            file_name = './result/hyper' + str(i) + '.pkl'
            open_file = open(file_name, "wb")
#            pickle.dump(sample_list, open_file)
            open_file.close()
            
            np.savetxt('./result/loss-mat.txt',np.array(loss_record),fmt='%10.5e')
            np.savetxt('./result/loss_ode1-mat.txt',np.array(loss_ode1_record),fmt='%10.5e')
            np.savetxt('./result/loss_u-mat.txt',np.array(loss_u_record),fmt='%10.5e')
            np.savetxt('./result/loss_ode2-mat.txt',np.array(loss_ode2_record),fmt='%10.5e')
            np.savetxt('./result/loss_v-mat.txt',np.array(loss_v_record),fmt='%10.5e')
            np.savetxt('./result/sigma1-mat.txt',np.array(sigma1_record),fmt='%10.5e')
            np.savetxt('./result/sigma2-mat.txt',np.array(sigma2_record),fmt='%10.5e')
            np.savetxt('./result/lambda1-mat.txt',np.array(lambda1_record),fmt='%10.5e')
            np.savetxt('./result/lambda2-mat.txt',np.array(lambda2_record),fmt='%10.5e')
            np.savetxt('./result/lambda3-mat.txt',np.array(lambda3_record),fmt='%10.5e')
            np.savetxt('./result/lambda4-mat.txt',np.array(lambda4_record),fmt='%10.5e') 
            
            np.savetxt('./result/u' + str(i) + '-mat.txt',np.array(ut0),fmt='%10.5e')
            np.savetxt('./result/v' + str(i) + '-mat.txt',np.array(vt0),fmt='%10.5e')
            
            np.savetxt('./result/u_opt-mat.txt',np.array(u_opt),fmt='%10.5e')
            np.savetxt('./result/v_opt-mat.txt',np.array(v_opt),fmt='%10.5e')


# In[ ]:




