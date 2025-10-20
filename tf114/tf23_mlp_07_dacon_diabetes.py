from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_random_seed(2222)

path = './backups/Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

train = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
train[zero_na_columns] = train[zero_na_columns].replace(0, np.nan)

x_data = train.fillna(train.mean())
y_data = train_csv['Outcome']

# print(x.shape, y.shape) #(652, 8) (652,)
y_data = y_data.to_numpy().reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, shuffle=True, random_state=123, stratify=y_data
)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([8, 32]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([32]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([32, 16]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([16]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([16,1]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.matmul(x, w1) + b1
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.matmul(drop1, w2) + b2
layer3 = tf.matmul(layer2, w3) + b3
hypo   = tf.sigmoid(layer3)

# 2) 수치안정한 BCE 로스 (log(0) 방지)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer3))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

epochs = 100000
patience = 0
threshold = np.inf
best_acc = 0
best_loss = 0
best_step=0
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs) :
        _, train_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train, dr:0.4})
        val_loss, y_prob = sess.run([loss, hypo], feed_dict={x:x_test, y:y_test, dr:0.})
        y_pred = np.round(y_prob)
        val_acc = accuracy_score(y_test.ravel(), y_pred.ravel())
        
        if step % 100 ==0 :
            print(f"{step+1}\tEpochs | Loss {train_loss:.6f}\t| Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")
            
        if val_loss < threshold :
            threshold = val_loss
            best_loss = val_loss
            best_acc = val_acc
            best_step = step+1
            patience=0
        else :
            patience+=1
            if patience==100:
                print(f"\nEarly Stopping Triggered - {step+1} epochs")
                print(f"Best Epochs {best_step}\t| Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")
                break
            
# Early Stopping Triggered - 3032 epochs
# Best Epochs 2932        | Val Loss 0.506890     | Val ACC 0.778626

# Early Stopping Triggered - 520 epochs
# Best Epochs 420 | Val Loss 6.016720     | Val ACC 0.664122