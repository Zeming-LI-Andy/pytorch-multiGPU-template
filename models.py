import torch
import torch.nn as nn

# TODO
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.act1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(512, 256)
        self.act2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(256, 64)
        self.act3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(64, 10)
    
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize some fixed parameter such as pos embedding
        
        # initialize torch Layer such as nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    # TODO
    """ initializing parameters """
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input):
        x, y = input
        out = torch.flatten(x, start_dim=1)

        out = self.linear1(out)
        out = self.act1(out)

        out = self.linear2(out)
        out = self.act2(out)

        out = self.linear3(out)
        out = self.act3(out)

        out = self.linear4(out)

        loss = torch.nn.functional.cross_entropy(out, y)
        return out, loss


def MyModel_versionID(**kwargs):
    print(kwargs)
    return MyModel(**kwargs)