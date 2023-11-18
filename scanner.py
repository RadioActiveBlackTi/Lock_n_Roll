import modi
import numpy as np
import time
from data_utils import dataset_enroll

def modi_connect():
    bundle = modi.MODI()
    button = bundle.buttons[0]
    motor = bundle.motors[0]
    gyro = bundle.gyros[0]
    env = bundle.envs[0]
    ir1 = bundle.irs[0]
    ir2 = bundle.irs[1]
    return bundle, button, motor, gyro, env, ir1, ir2

def scan_key(range_num, modis):
    if 100%range_num:
        raise ValueError("100%range_num should be 0")

    bundle, button, motor, gyro, env, ir1, ir2 = modis

    scanned_data = []

    motor.degree = 0, 0
    time.sleep(3)
    for i in range(1,range_num+1):
        prox1, prox2 = ir1.proximity, ir2.proximity
        r, g, b = env.red, env.green, env.blue
        motor.degree = (100//range_num)*i, (100//range_num)*i
        time.sleep(0.1)
        velx, vely, velz = gyro.angular_vel_x, gyro.angular_vel_y, gyro.angular_vel_z
        scan = [prox1, prox2, r, g, b, velx, vely, velz]
        scanned_data.append(scan)
        time.sleep(1)

    scanned_data = np.array(scanned_data)[1:,:]
    print(scanned_data)
    motor.degree = 0, 0
    time.sleep(3)

    return scanned_data

if __name__ == "__main__":
    mode = 'dataset'
    range_num = 10
    modis = modi_connect()
    if mode=='dataset':
        key = int(input("Dataset Enroll >> Please write key index: "))
        while(True):
            scan = scan_key(range_num, modis)
            dataset_enroll("./dataset.csv", scan, key)
            if input("Dataset Enroll >> Continue?: ")=='q':
                break
    elif mode=='key':
        scan = scan_key(range_num, modis)
        key = int(input("Key Enroll >> Please write key index: "))
        dataset_enroll("./key.csv", scan, key)
    else:
        scan = scan_key(range_num, modis)
        print(scan)