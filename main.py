import torch
import pandas as pd
import keyboard

import scanner
import safe
from model import Discriminator
from data_utils import dataset_enroll, dataset_extract

device = "cuda" if torch.cuda.is_available() else "cpu"

NoButton = True
uuid = '96BA8B36'
threshold = 0.75

if __name__ == "__main__":
    sc = scanner.Scanner()
    lock = safe.Safe(uuid)

    net = Discriminator(8, 128, 8)
    net.load_state_dict(torch.load("./resources/model_state_dict.pt", map_location=torch.device(device)))
    net.to(device)
    net.eval()

    lock.mode = 1 # 1: inference mode, 0: enroll mode
    print("="*32)
    print("<< Commands >>")
    print("q: Quit Lock'n Roll")
    print("u: Explicitly Unlock the safe")
    print("l: Explicitly lock the safe")
    print("m: Toggle mode")
    print("s: Select Key")
    print("=" * 32)
    while (True):
        k = keyboard.read_key()
        if k == "q":
            print(">> Exiting Lock'n Roll...")
            break

        elif k=="u":
            print(">> Unlocking Safe Explicitly...")
            lock.unlock_explicit()

        elif k=="l":
            print(">> Locking Safe Explicitly...")
            lock.lock_explicit()

        elif k=="m":
            if lock.mode:
                print(">> Changing to Enroll Mode")
                lock.mode = 0
                lock.set_text()
            else:
                print(">> Changing to Inference Mode")
                lock.mode = 1
                lock.set_text()

        elif k=="s":
            lock.select()
            print(f">> Key Selected: {lock.key}")

        if sc.button.pressed:
            if not lock.state and lock.mode:
                lock.lock_explicit()
            else:
                lock.display.text = "Scanning..."
                scanned = sc.scan_key(10)
                key = lock.key
                if not lock.mode:
                    print(">> Scanned Data")
                    print(scanned)
                    dataset_enroll("./resources/key.csv", scanned, key, enroll_key=True)
                else:
                    key_data = pd.read_csv("./resources/key.csv")
                    key_scan, _ = dataset_extract(key_data, key, key_finding=True)
                    scanned = torch.FloatTensor(scanned).unsqueeze(0)
                    key_scan = torch.FloatTensor(key_scan).unsqueeze(0)

                    prob = net(scanned, key_scan)
                    print(f">> Probability of same: {float(prob): .2f}")
                    lock.locker(prob, threshold)
                lock.set_text()

