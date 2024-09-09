# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole_gnn.recbole_gnn.model.general_recommender import SGL, NCL
from recbole.model.general_recommender import NGCF, LightGCN

class GraphGenerator_VAE(nn.Module):
    def __init__(self, emb_size):
        super(GraphGenerator_VAE, self).__init__()
        self.latent_size = emb_size
        self.encoder = nn.Linear(self.latent_size * 2, 64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc_encoder = nn.Linear(64, 64)
        self.fc_encoder_mu = nn.Linear(64, 16)
        self.fc_encoder_var = nn.Linear(64, 16)
        self.fc_reparameterize = nn.Linear(16, 64)
        self.fc_decode = nn.Linear(64, 1)

    def encode(self, x):
        output = self.encoder(x)
        h = self.relu(output)
        # return self.fc_encoder_mu(h), self.fc_encoder_var(h)
        return self.fc_encoder(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        # h = self.relu(self.fc_reparameterize(z))
        return self.sigmoid(self.fc_decode(z))

    def forward(self, user_e, item_e):
        input_vec = torch.cat((user_e, item_e), axis=1)
        # input_vec = user_e+item_e
        # mu, log_var = self.encode(input_vec)
        # z = self.reparameterize(mu, log_var)
        z = self.encode(input_vec)
        return self.decode(z)


class BasePairDenoiseCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(BasePairDenoiseCF, self).__init__(config, dataset)
        self.device = config["device"]
        self.backbone = backbone
        if isinstance(self.backbone, NCL):
            self.e_step = self.backbone.e_step()

    def forward(self, sgl_graph=None):
        if isinstance(self.backbone, SGL):
            ret = self.backbone.forward(graph=sgl_graph)
        else:
            ret = self.backbone.forward()
        return ret

    def calculate_loss(self, interaction):
        return self.backbone.calculate_loss(interaction)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.backbone.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.backbone.restore_user_e is None or self.backbone.restore_item_e is None:
            self.backbone.restore_user_e, self.backbone.restore_item_e = self.backbone.forward()
        # get user embedding from storage variable
        u_embeddings = self.backbone.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.backbone.restore_item_e.transpose(0, 1))
        return scores.view(-1)


class BasePointDenoiseCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset, backbone):
        super(BasePointDenoiseCF, self).__init__(config, dataset)
        self.device = config["device"]
        self.LABEL = config["LABEL_FIELD"]
        self.backbone = backbone
        if isinstance(self.backbone, NCL):
            self.e_step = self.backbone.e_step()

    def calculate_loss(self, interaction):
        return self.backbone.calculate_loss(interaction)

    def forward(self, sgl_graph=None):
        if isinstance(self.backbone, SGL):
            ret = self.backbone.forward(graph=sgl_graph)
        else:
            ret = self.backbone.forward()
        return ret
