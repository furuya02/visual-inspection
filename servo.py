import Jetson.GPIO as GPIO
from sensor import Sensor
import time


class Servo:
    VERVO_PIN = 18

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)  # 物理ピン番号で指定
        GPIO.setup(self.VERVO_PIN, GPIO.OUT, initial=GPIO.HIGH)
        self.p1 = GPIO.PWM(self.VERVO_PIN, 50)

        self.p1.start(9)
        # self.p1.ChangeDutyCycle(9)

    def close_gate(self):
        self.p1.ChangeDutyCycle(5)
        time.sleep(0.5)
        self.p1.ChangeDutyCycle(9)
