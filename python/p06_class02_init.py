class Father:                   # ()이게 없으면 상속 받을 게 없다는 뜻 절대 함수처럼 변수 아님
    def __init__(self, name):   # 인스턴스를 생성하면 자동으로 실행됨 ()여기에서 이거는 변수 맞음
        self.name = name
        print("> Father__init__ 실행")
        print(self.name, '아빠')
        print("> Father__init__ 끝")
    babo = 4

class Mother:
    def __init__(self, name):
        self.name = name
        print("> Mother__init__ 실행")
        print(self.name, '엄마')
        print("> Mother__init__ 끝")
    mutter = 10

class Son(Father, Mother):
    def __init__(self, name):
        print("> Son__init__ 실행")
        super().__init__(name)
        print('바보 + 천재 :', self.babo + self.cheonje)
        print('바보 + 천재 :', super().babo + self.cheonje)
        print("> Son__init__ 끝")
    cheonje = 5

bbb = Son('흥민')
'''
bbb = Son.babo
aaa = Son.cheonje
ccc = Son.mutter

print(bbb)
print(aaa)
print(aaa+bbb)
print(ccc)

ddd = Son('재현')
'''