import modi
import numpy as np
import time
from data_utils import dataset_enroll


class Scanner:
    def __init__(self):
        bundle = modi.MODI()
        self.bundle = bundle
        self.button = bundle.buttons[0]
        self.motor = bundle.motors[0]
        self.gyro = bundle.gyros[0]
        self.env = bundle.envs[0]
        self.ir1 = bundle.irs[0]
        self.ir2 = bundle.irs[1]

    def scan_key(self, range_num):
        if 100%range_num:
            raise ValueError("100%range_num should be 0")

        scanned_data = []

        self.motor.degree = 0, 0
        time.sleep(3)
        for i in range(1,range_num+1):
            prox1, prox2 = self.ir1.proximity, self.ir2.proximity
            r, g, b = self.env.red, self.env.green, self.env.blue
            self.motor.degree = (100//range_num)*i, (100//range_num)*i
            time.sleep(0.1)
            velx, vely, velz = self.gyro.angular_vel_x, self.gyro.angular_vel_y, self.gyro.angular_vel_z
            scan = [prox1, prox2, r, g, b, velx, vely, velz]
            scanned_data.append(scan)
            time.sleep(1)

        scanned_data = np.array(scanned_data)[1:,:]
        self.motor.degree = 0, 0
        time.sleep(3)

        return scanned_data


if __name__ == "__main__":
    mode = 'key'
    range_num = 10
    scanner = Scanner()
    if mode=='dataset':
        key = int(input("Dataset Enroll >> Please write key index: "))
        while(True):
            scan = scanner.scan_key(range_num)
            print(scan)
            dataset_enroll("./resources/dataset.csv", scan, key)
            if input("Dataset Enroll >> Continue?: ")=='q':
                break
    elif mode=='key':
        scan = scanner.scan_key(range_num)
        print(scan)
        key = int(input("Key Enroll >> Please write key index: "))
        dataset_enroll("./resources/key.csv", scan, key, enroll_key=True)
    else:
        scan = scanner.scan_key(range_num)
        print(scan)