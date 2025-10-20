import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
tf.compat.v1.random.set_random_seed(777)

(x_train_data, y_train_data), (x_test_data, y_test_data) = tf.keras.datasets.boston_housing.load_data()

y_train_data = y_train_data.reshape(-1,1)
y_test_data = y_test_data.reshape(-1,1)

print(x_train_data.shape, "|", y_train_data.shape)    # (404, 13) | (404,)
print(x_test_data.shape, "|", y_test_data.shape)      # (102, 13) | (102,)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([13, 32]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([32]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([32, 16]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([16]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([16, 1]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.matmul(x_train, w1) + b1
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.matmul(drop1, w2) + b2
hypo = tf.matmul(layer2, w3) + b3

loss = tf.reduce_mean(tf.square(hypo - y_train))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

epochs=1000000
threshold = np.inf
counts = 0
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs):
        _, trnloss= sess.run([train, loss],
                    feed_dict = {
                        x_train:x_train_data, y_train: y_train_data, dr:0.3
                        }
                    )
        val_loss, y_pred = sess.run([loss, hypo], feed_dict={
                        x_train:x_test_data, y_train: y_test_data, dr:0.3
                        })
        val_rmse = np.sqrt(mean_squared_error(y_test_data, y_pred))
        val_r2 = r2_score(y_test_data, y_pred)
        
        if step % 100 == 0 :
            print(step+1, f"\tepochs | Loss {trnloss:.6f}\t| R2 {val_r2:.6f}\t| RMSE {val_rmse:.6f}")
        
        if val_loss < threshold :
            threshold = val_loss
            best_steps = step
            best_rmse = val_rmse
            best_r2 = val_r2
            
            counts = 0
        else :
            counts += 1
            
            if counts == 200 :
                print(f"\nEarly Stopping Triggered - {step+1} epoch")
                print("Best Step", best_steps+1, f"\tepochs | RMSE {best_rmse:.6f}\t| R2 {best_r2:.6f}")
                break
            else :
                continue
        
