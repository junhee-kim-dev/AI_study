import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
tf.random.set_random_seed(5555)

path = './backups/Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = LabelEncoder()
oe.fit(train_csv['Geography'])
train_csv['Geography'] = oe.transform(train_csv['Geography'])
test_csv['Geography'] = oe.transform(test_csv['Geography'])

oe_g = LabelEncoder()
oe_g.fit(train_csv['Gender'])
train_csv['Gender'] = oe_g.transform(train_csv['Gender'])
test_csv['Gender'] = oe_g.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x_data = train_csv.drop(['Exited'], axis=1)
y_data = train_csv['Exited']

print(x_data.shape)

y_data = y_data.to_numpy().reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.9, shuffle=True, random_state=123, stratify=y_data
)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([10, 32]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([32]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([32, 16]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([16]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([16,1]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.nn.relu(tf.matmul(drop1, w2) + b2)
layer3 = tf.matmul(layer2, w3) + b3
hypo   = tf.sigmoid(layer3)

# 2) 수치안정한 BCE 로스 (log(0) 방지)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer3))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

epochs=10000
patience=0
best_acc = 0
best_loss = 0
best_step = 0
threshold = np.inf

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
            if patience==200:
                print(f"\nEarly Stopping Triggered - {step+1} epochs")
                print(f"Best Epochs {best_step}\t| Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")
                break


# Early Stopping Triggered - 274 epochs
# Best Epochs 174 | Val Loss 148.075668   | Val ACC 0.788415

# Early Stopping Triggered - 699 epochs
# Best Epochs 499 | Val Loss 152.718933   | Val ACC 0.787930