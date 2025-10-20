from machine.car import drive
from machine.tv import watch

drive()
watch()

print("="*20)
from machine import car
from machine import tv
car.drive()
tv.watch()

print("="*20)
from machine import car, tv
car.drive()
tv.watch()

print("="*10, "test", "="*10)
from machine.test.car import drive
from machine.test.tv import watch
drive()
watch()

print("="*10, "test", "="*10)
from machine.test import car, tv
car.drive()
tv.watch()

print("="*10, "test", "="*10)
from machine import test
test.car.drive()
test.tv.watch()
