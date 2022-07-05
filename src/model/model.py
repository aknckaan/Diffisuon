from torch import nn
import torch
from src.model.layers import build_layers, conv2d_norm

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_layers, self.up_layers = build_layers([1, 1, 1], 8)
        self.first = conv2d_norm(3, 8, 7, 1)
        self.last = conv2d_norm(8, 3, 7, 1)


    def forward(self, x):
        cur_x = self.first(x)
        x_d_arr = []


        for res_layers, compress in self.down_layers:
            cur_x = res_layers(cur_x)
            cur_x = compress(cur_x)
            x_d_arr.append(cur_x)

        # x_d_arr.append(None)

        for (res_layers, expand), res_x in zip(self.up_layers, x_d_arr[::-1]):
            cur_x = res_layers(cur_x + res_x)
            cur_x = expand(cur_x)

        x = self.last(cur_x)
        
        return x            
        

if __name__  == "__main__":
    unet = UNET()
    t = torch.zeros([1, 3, 128, 128])
    print(unet(t).shape)
