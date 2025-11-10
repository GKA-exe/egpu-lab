import Jetson. GPIO as GPIO 
import time 
 
button_pin = 17 
led_pin = 18

GPIO.setmode(GPIO.BCM) 
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(led_pin, GPIO.OUT) 
 
try: 
    while True: 
        button_state = GPIO.input (button_pin) 
        if button_state == GPIO.LOW: 
            GPIO.output (led_pin, GPIO.HIGH) 
        else: 
            GPIO.output (led_pin, GPIO.LOW) 
        time.sleep(0.1) 
 
except KeyboardInterrupt: 
    pass 
finally: 
    GPIO.cleanup() 
