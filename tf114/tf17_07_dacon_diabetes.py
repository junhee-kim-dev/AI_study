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


x = tf.placeholder(tf.float32, shape=[None,8])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([8,1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

logits = tf.matmul(x,w) + b
hypo = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
        _, train_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
        val_loss, y_prob = sess.run([loss, hypo], feed_dict={x:x_test, y:y_test})
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

