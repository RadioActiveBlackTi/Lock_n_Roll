import modi
import time

unlock_angle = 50
unlock_tune = 800
lock_tune = 600
volume = 70


class Safe:
    def __init__(self, uuid, initial_state=1):
        # tentative
        bundle = modi.MODI(conn_type='ble', network_uuid=uuid)
        self.bundle = bundle
        self.button = bundle.buttons[0]
        self.dial = bundle.dials[0]
        self.motor = bundle.motors[0]
        self.led = bundle.leds[0]
        self.speaker = bundle.speakers[0]
        self.display = bundle.displays[0]

        self.display.text = "Lock'n Roll"
        self.motor.degree = (0, 0) if initial_state else (unlock_angle,0)

    def locker(self, value, threshold):
        if value >= threshold:
            self.motor.degree = unlock_angle, 0  # tentative
            self.display.text = "Unlocked!"
            self.speaker.tune = unlock_tune, volume  # tentative
            self.led.rgb = 0, 100, 0
        else:
            self.motor.degree = 0, 0 # tentative
            self.display.text = f"Not Unlocked\nSimilarity: {value * 100:.2f}%"
            self.speaker.tune = lock_tune, volume
            self.led.rgb = 100, 0, 0
        time.sleep(2)
        self.led.rgb = 0, 0, 0
        self.speaker.tune = 0, 0
        self.display.text = "Lock'n Roll"

    def select(self):
        value_mapper = [i for i in range(10)]
        key_idx = value_mapper[self.dial.degree//10]
        self.display.text = f"Selected Key: {key_idx}"
        time.sleep(2)
        self.display.text = "Lock'n Roll"

    def unlock_explicit(self):
        self.motor.degree = unlock_angle, 0  # tentative
        self.display.text = "Unlocked Explicitly"
        time.sleep(2)
        self.display.text = "Lock'n Roll"

    def lock_explicit(self):
        self.motor.degree = 0, 0  # tentative
        self.display.text = "Locked Explicitly"
        time.sleep(2)
        self.display.text = "Lock'n Roll"


if __name__ == "__main__":
    threshold = 0.5
    uuid = 0
    safe = Safe(uuid)
    while(True):
        k = input("Type Value or Command (To quit, type q) >> ")
        if k=='q':
            break
        elif k=='u':
            safe.unlock_explicit()
        elif k=='l':
            safe.lock_explicit()
        elif k=='s':
            safe.select()
        else:
            try:
                value = float(k)
            except:
                print('Please Type Properly.')
                continue
            safe.locker(value, threshold)
