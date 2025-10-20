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

w = tf.Variable(tf.random.normal([10,1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

logits = tf.matmul(x, w) + b
hypo = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
        _, train_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
        val_loss, y_prob = sess.run([loss, hypo], feed_dict={x:x_test, y:y_test})
        y_pred = np.round(y_prob)
        val_acc = accuracy_score(y_test.ravel(), y_pred.ravel())
        
        if step % 50 ==0 :
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


# Early Stopping Triggered - 274 epochs
# Best Epochs 174 | Val Loss 148.075668   | Val ACC 0.788415