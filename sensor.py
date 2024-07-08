import Jetson.GPIO as GPIO


class Sensor:
    SENSOR_PIN = 16
    mode = "ready"  # ready | sleep
    status = "off"  # on | off

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)  # 物理ピン番号で指定
        GPIO.setup(self.SENSOR_PIN, GPIO.IN)

    def reset(self):
        self.mode = "ready"
        self.status = "off"

    def check(self):
        sensor = GPIO.input(self.SENSOR_PIN)

        if self.status == "off" and self.mode == "ready":
            sensor = GPIO.input(self.SENSOR_PIN)
            if sensor == 0:
                self.status = "on"
                self.mode = "sleep"  # resetされるまでは、現在のstatusを維持する

        return self.status
