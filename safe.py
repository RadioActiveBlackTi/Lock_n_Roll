import modi
import time

unlock_angle = 50
unlock_tune = 800
lock_tune = 600
volume = 70


class Safe:
    def __init__(self, uuid, initial_state=1):
        # tentative
        if uuid:
            bundle = modi.MODI(conn_type='ble', network_uuid=uuid)
        else:
            bundle = modi.MODI()
        self.bundle = bundle
        self.dial = bundle.dials[0]
        self.motor = bundle.motors[0]
        self.led = bundle.leds[0]
        self.speaker = bundle.speakers[0]
        self.display = bundle.displays[0]

        self.mode = 1 # 1: Inference Mode, 0: Enroll Mode
        self.set_text()

        self.state = initial_state # 1 for locked, 0 for unlocked
        self.motor.degree = (0, 0) if initial_state else (unlock_angle,0)
        self.key_idx = self.select()

    def locker(self, value, threshold):
        if value >= threshold:
            self.motor.degree = 0, unlock_angle  # tentative
            self.display.text = "Unlocked!"
            self.state = 0
            self.speaker.tune = unlock_tune, volume  # tentative
            self.led.rgb = 0, 100, 0
        else:
            self.motor.degree = 0, 0 # tentative
            self.display.text = f"Not Unlocked\nSimilarity: {float(value) * 100:.2f}%"
            self.state = 1
            self.speaker.tune = lock_tune, volume
            self.led.rgb = 100, 0, 0
        time.sleep(2)
        self.led.rgb = 0, 0, 0
        self.speaker.tune = 0, 0
        self.set_text()

    def select(self):
        value_mapper = [i for i in range(10)]
        if self.dial.degree==100:
            key_idx = value_mapper[9]
        else:
            key_idx = value_mapper[int(self.dial.degree)//10]
        self.display.text = f"Selected Key: {key_idx}"
        time.sleep(2)
        self.set_text()
        self.key = key_idx
        return key_idx

    def unlock_explicit(self):
        self.motor.degree = 0, unlock_angle  # tentative
        self.display.text = "Unlocked Explicitly"
        self.state = 0
        time.sleep(2)
        self.set_text()

    def lock_explicit(self):
        self.motor.degree = 0, 0  # tentative
        self.display.text = "Locked Explicitly"
        self.state = 1
        time.sleep(2)
        self.set_text()

    def set_text(self):
        if self.mode:
            self.display.text = "Lock'n Roll"
        else:
            self.display.text = "Enroll Mode"


if __name__ == "__main__":
    threshold = 0.5
    uuid = '96BA8B36'
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
            print(safe.key)
        else:
            try:
                value = float(k)
            except:
                print('Please Type Properly.')
                continue
            safe.locker(value, threshold)
