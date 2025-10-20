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
