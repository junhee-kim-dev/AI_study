x = 10
y = 10
w = 0.001

lr = 0.001
epoch= 1000

for i in range(epoch) :
    hypo = x*w
    loss = (hypo - y) **2
    print(f"Loss {round(loss,4)} \t Predict {round(hypo, 4)}")
    
    up_predict = x*(w+lr)
    up_loss = (y-up_predict) **2
    
    down_predict = x*(w-lr)
    down_loss = (y - down_predict) **2
    
    if (up_loss > down_loss) :
        w = w-lr
    else :
        w = w+lr