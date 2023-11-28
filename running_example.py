import torch
import model
from scanner import Scanner

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    net = model.Discriminator(8, 128, 8)
    net.load_state_dict(torch.load("./resources/model_state_dict.pt", map_location=torch.device(device)))
    net.to(device)
    net.eval()

    scanner = Scanner()
    input("Continue to type: ")
    x = scanner.scan_key(10)
    input("Continue to type: ")
    y = scanner.scan_key(10)
    print(net(torch.FloatTensor(x).unsqueeze(0), torch.FloatTensor(y).unsqueeze(0)))