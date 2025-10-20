class Father:   # ()이게 없으면 상속 받을 게 없음
    def __init__(self, name):   # 인스턴스를 생성하면 자동으로 실행됨
        self.name = name
        print("> Father__init__ 실행")
        print(self.name, '아빠')
        print("> Father__init__ 끝")
    patter = 4


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
        super().patter
        print("> Son__init__ 끝")

aaa = Father('재현')
print('='*10)

ccc = Father.patter
print(ccc)
print('='*10)

bbb = Son('흥민')
print('='*10)

patter = Son.patter
print(patter)
print('='*10)

mutter = Son.mutter
print(mutter)







