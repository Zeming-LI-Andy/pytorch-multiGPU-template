import torch
import torch.nn as nn

# TODO
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()

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

    def forward_encoder(self):
        pass

    def forward_decoder(self):
        pass

    def forward_loss(self):
        pass

    def forward(self):
        pass


def MyModel_versionID(**kwargs):
    print(kwargs)
    return MyModel(**kwargs)