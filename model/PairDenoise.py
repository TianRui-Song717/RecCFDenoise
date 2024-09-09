# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.utils import InputType
from recbole_gnn.recbole_gnn.model.general_recommender import SGL, NCL
from recbole.model.general_recommender import NGCF, LightGCN
from model.base import BasePairDenoiseCF


class TCEPairDenoise(BasePairDenoiseCF):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(TCEPairDenoise, self).__init__(config, dataset, backbone)
        self.bpr_gamma = 1e-10
        self.count = 0
        self.exponent = config["TCE_exponent"]
        self.num_update = config["TCE_num_update"]              # config["TCE_num_update"]
        self.max_drop_rate = config["TCE_max_drop_rate"]        # config["TCE_max_drop_rate"]

    def _get_drop_rate(self):
        drop_rate = torch.linspace(
            0, self.max_drop_rate ** self.exponent, self.num_update
        )
        if self.count < self.num_update:
            ret = drop_rate[self.count]
        else:
            ret = self.max_drop_rate ** self.exponent
        self.count += 1  # update drop rate per batch
        return ret

    def tce_denoise(self, pos_scores, neg_scores):
        bsz = pos_scores.size(0)
        remain_rate = 1 - self._get_drop_rate()
        remain_num = int(remain_rate * bsz)
        with torch.no_grad():
            batch_loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores))  # [bsz, ]
            idxs_sorted = torch.argsort(batch_loss, descending=False)
            idxs_update = idxs_sorted[:remain_num]
        loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores[idxs_update] - neg_scores[idxs_update])).mean()
        return loss

    def calculate_loss(self, interaction):
        if self.backbone.restore_user_e is not None or self.backbone.restore_item_e is not None:
            self.backbone.restore_user_e, self.backbone.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Model Forward
        user_all_embeddings, item_all_embeddings = self.backbone.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]


        if isinstance(self.backbone, LightGCN):
            u_ego_embeddings = self.backbone.user_embedding(user)
            pos_ego_embeddings = self.backbone.item_embedding(pos_item)
            neg_ego_embeddings = self.backbone.item_embedding(neg_item)
            reg_loss = self.backbone.reg_loss(
                u_ego_embeddings,
                pos_ego_embeddings,
                neg_ego_embeddings,
                require_pow=self.backbone.require_pow,
            )

            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            rec_loss = self.tce_denoise(pos_scores, neg_scores)
            loss = rec_loss + self.backbone.reg_weight * reg_loss
        elif isinstance(self.backbone, NGCF):
            reg_loss = self.reg_loss(
                u_embeddings, pos_embeddings, neg_embeddings
            )

            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            rec_loss = self.tce_denoise(pos_scores, neg_scores)
            loss = rec_loss + self.reg_weight * reg_loss
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}'s 'calculate_loss' function for backbone '{self.backbone.__class__.__name__}' is not implemented!"
            )

        return loss
