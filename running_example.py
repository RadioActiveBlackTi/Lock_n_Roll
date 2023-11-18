import torch
import model
import data_collect

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    net = model.Discriminator(8, 128, 8)
    net.load_state_dict(torch.load("./model_state_dict.pt"))
    net.to(device)
    net.eval()

    modis = data_collect.modi_connect()
    input("Continue to type: ")
    x = data_collect.scan_key(10, modis)
    input("Continue to type: ")
    y = data_collect.scan_key(10, modis)
    print(net(torch.FloatTensor(x).unsqueeze(0), torch.FloatTensor(y).unsqueeze(0)))