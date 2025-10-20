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

w = tf.compat.v1.Variable(tf.random.normal([13, 1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(0, dtype=tf.float32)
lr = tf.compat.v1.Variable(0.1, dtype=tf.float32)

hypo = tf.matmul(x_train, w) + b

loss = tf.reduce_mean(tf.square(hypo - y_train))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)

epochs=1000000
threshold = np.inf
counts = 0
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs):
        _, val_w, val_b = sess.run([train, w, b],
                    feed_dict = {
                        x_train:x_train_data, y_train: y_train_data
                        }
                    )
        val_hypo = np.matmul(x_test_data, val_w) + val_b
        val_loss = mean_squared_error(val_hypo,y_test_data)
        val_rmse = np.sqrt(val_loss)
        val_r2 = r2_score(val_hypo,y_test_data)
        val_mae = mean_absolute_error(val_hypo,y_test_data)
        
        if step % 100 == 0 :
            print(step+1, f"\tepochs | RMSE {val_rmse:.6f}\t| R2 {val_r2:.6f}\t| MAE {val_mae:.6f}")
        
        if val_loss < threshold :
            threshold = val_loss
            best_steps = step
            best_rmse = val_rmse
            best_mae = val_mae
            best_r2 = val_r2
            
            best_w = val_w
            best_b = val_b
            counts = 0
        else :
            counts += 1
            
            if counts == 50 :
                print(f"\nEarly Stopping Triggered - {step+1} epoch")
                print("Best Step", best_steps+1, f"\tepochs | RMSE {best_rmse:.6f}\t| R2 {best_r2:.6f}| MAE {best_mae:.6f}")
                break
            else :
                continue
        

# y_pred = np.matmul(x_test_data, best_w) + best_b

# r2 = r2_score(y_test_data, y_pred)
# mae = mean_absolute_error(y_test_data, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test_data, y_pred))
# print(f"RMSE {rmse:.6f}\t| R2 {r2:.6f}\t| MAE {mae:.6f}\t")

