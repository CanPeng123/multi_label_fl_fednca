import torch.nn as nn
from collections import OrderedDict


class FCClassifierLayer(nn.Module):
    def __init__(self, encoder_dim, num_images, num_labels, norm_type="batch_norm"):
        super(FCClassifierLayer, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_images = num_images
        self.num_labels = num_labels
        print("FCClassifierLayer | num_images: {0}, num_labels: {1}, norm_type: {2}".format(num_images, num_labels, norm_type))

        self.clf_layer = nn.Sequential(
            OrderedDict([
                ("clf_fc0", nn.Linear(self.encoder_dim * self.num_images, self.encoder_dim * 2)),
                ("clf_norm0", nn.LayerNorm(self.encoder_dim * 2)),
                ("clf_actv0", nn.GELU()),
                ("clf_fc1", nn.Linear(self.encoder_dim * 2, self.num_labels))
            ])
        )
        

    def forward(self, x):
        return self.clf_layer(x)
