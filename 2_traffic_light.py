import Jetson.GPIO as GPIO
import time
red_pin = 7
yellow_pin = 11
green_pin = 13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(red_pin, GPIO.OUT)
GPIO.setup(yellow_pin, GPIO.OUT)
GPIO.setup(green_pin, GPIO.OUT)
try:
    while True:
        GPIO.output(red_pin, GPIO.HIGH)
        GPIO.output(yellow_pin, GPIO.LOW)
        GPIO.output(green_pin, GPIO.LOW)
        time.sleep(3)
        
        GPIO.output(red_pin, GPIO.HIGH)
        GPIO.output(yellow_pin, GPIO.HIGH)
        GPIO.output(green_pin, GPIO.LOW)
        time.sleep(1)
        
        GPIO.output(red_pin, GPIO.LOW)
        GPIO.output(yellow_pin, GPIO.LOW)
        GPIO.output(green_pin, GPIO.HIGH)
        time.sleep(5)
        
        GPIO.output(red_pin, GPIO.LOW)
        GPIO.output(yellow_pin, GPIO.HIGH)
        GPIO.output(green_pin, GPIO.LOW)
        time.sleep(2)
except KeyboardInterrupt:
    GPIO.cleanup()
