#1.
aaa = [1,2,3,4,5]
for i in aaa:
    print(i)    # epochs=100 같은 게 100번 반복하는 거임
# 1
# 2
# 3
# 4
# 5

#2.
add = 0
for i in range(1,101):
    add = add + i
print(add)
# 5050

#3.
results = []
for i in aaa:
    results.append(i+1)
print(results)
# [2, 3, 4, 5, 6]

# 오늘의 python : 
# for문과 list의 append