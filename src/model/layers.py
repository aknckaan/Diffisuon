from torch import nn
import torch

class conv2d_res(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv2d = nn.Sequential(conv2d_norm(in_features, in_features//2, 1, 1),
                                    conv2d_norm(in_features//2, in_features//2, 3, 1),
                                    conv2d_norm(in_features//2, in_features, 1, 1))
    def forward(self, x):
        x_ = self.conv2d(x)

        return x+x_


class conv2d_norm(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride):
        super().__init__()

        self.conv2d = nn.Conv2d(in_features, out_features, kernel, stride, (kernel-1)//2)
        self.act = nn.ReLU(1e-3)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.act(self.bn(self.conv2d(x)))

        return x

class conv2d_compress(nn.Module):
    def __init__(self, in_features, kernel):
        super().__init__()
        self.conv2d = conv2d_norm(in_features, in_features*2, kernel, 2)
    def forward(self, x):
        x = self.conv2d(x)
        return x

class conv2d_norm_t(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride):
        super().__init__()
        self.conv2d = nn.ConvTranspose2d(in_features, out_features, kernel, 2, (kernel-1)//2, 1)
        self.act = nn.ReLU(1e-3)
        self.bn = nn.BatchNorm2d(out_features)
    def forward(self, x):
        x = self.act(self.bn(self.conv2d(x)))

        return x

class conv2d_expand(nn.Module):
    def __init__(self, in_features, kernel):
        super().__init__()
        self.conv2d = conv2d_norm_t(in_features, in_features//2, kernel, 2)
    def forward(self, x):
        x = self.conv2d(x)
        return x



def build_layers(reps, in_features):

    down_layers = []
    up_layers = []
    cur_features = in_features

    for rep in reps:
        res_layers = []
        up_res_layers = []
        for i in range(rep):
            res_layers.append(conv2d_res(cur_features))
            up_res_layers.append(conv2d_res(cur_features*2))

        res_layers_seq = nn.Sequential(*res_layers)
        up_res_layers_seq = nn.Sequential(*up_res_layers)

        down_layers.append([res_layers_seq, conv2d_compress(cur_features, 3)])
        up_layers.append([up_res_layers_seq, conv2d_expand(cur_features*2, 3)])

        cur_features = cur_features*2

    return down_layers, up_layers[::-1] [:-1]# remove the last one


if __name__ == "__main__":

    t = torch.zeros(1, 4, 64, 64)

    # conv2d_compress test
    x = conv2d_compress(4, 3)(t)
    assert x.shape == torch.Size([1, 8, 32, 32])
    #

    # conv2d_compress_t test
    x = conv2d_expand(4, 3)(t)
    assert x.shape == torch.Size([1, 2, 128, 128])
    #

    # conv2d_res test
    x = conv2d_res(4)(t)
    assert x.shape == torch.Size([1, 4, 64, 64])
    #

    # test build layers
    layers = build_layers([1, 1, 1], 32)
    print(layers)
    #