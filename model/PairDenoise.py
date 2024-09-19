# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.utils import InputType
from recbole_gnn.recbole_gnn.model.general_recommender import SGL, NCL
from recbole.model.general_recommender import NGCF, LightGCN
from model.base import BasePairDenoiseCF


class LossBasePairDenoise(BasePairDenoiseCF):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(LossBasePairDenoise, self).__init__(config, dataset, backbone)
        self.bpr_gamma = 1e-10

    def _denoise_loss(self, pos_scores, neg_scores, **kwargs):
        raise NotImplementedError("The denoise loss function is not implemented!")

    def calculate_loss(self, interaction, **kwargs):
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
            rec_loss = self._denoise_loss(pos_scores, neg_scores, **kwargs)
            loss = rec_loss + self.backbone.reg_weight * reg_loss
        elif isinstance(self.backbone, NGCF):
            reg_loss = self.reg_loss(
                u_embeddings, pos_embeddings, neg_embeddings
            )

            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            rec_loss = self._denoise_loss(pos_scores, neg_scores, **kwargs)
            loss = rec_loss + self.reg_weight * reg_loss
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}'s 'calculate_loss' function for backbone '{self.backbone.__class__.__name__}' is not implemented!"
            )

        return loss

class TCEPairDenoise(LossBasePairDenoise):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(TCEPairDenoise, self).__init__(config, dataset, backbone)
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

    def _denoise_loss(self, pos_scores, neg_scores):
        """
        Paper: Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua. Denoising Implicit Feedback for Recommendation.WSDM2021
        implemented according to https://github.com/WenjieWWJ/DenoisingRec/blob/main/T_CE/loss.py
        :param pos_scores: positive item prediction score
        :param neg_scores: negative item prediction score
        :return: loss value
        """
        bsz = pos_scores.size(0)
        remain_rate = 1 - self._get_drop_rate()
        remain_num = int(remain_rate * bsz)
        with torch.no_grad():
            batch_loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores))  # [bsz, ]
            idxs_sorted = torch.argsort(batch_loss, descending=False)
            idxs_update = idxs_sorted[:remain_num]
        if not isinstance(self.backbone, SGL):
            loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores[idxs_update] - neg_scores[idxs_update])).mean()
        else:
            loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores[idxs_update] - neg_scores[idxs_update])).sum()
        return loss


class RCEPairDenoise(LossBasePairDenoise):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(RCEPairDenoise, self).__init__(config, dataset, backbone)
        self.alpha = config["RCE_alpha"]

    def _denoise_loss(self, pos_scores, neg_scores):
        with torch.no_grad():
            loss_ = torch.sigmoid(-torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores)))
            weight = torch.pow(loss_, self.alpha)
        loss = weight * -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores))
        if not isinstance(self.backbone, SGL):
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss


class BODPairDenoise(LossBasePairDenoise):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, backbone):
        super(BODPairDenoise, self).__init__(config, dataset, backbone)

    def _denoise_loss(self, pos_scores, neg_scores, bod_weight_pos=None, bod_weight_neg=None):
        assert bod_weight_neg is not None and bod_weight_pos is not None, "BOD Weight is Empty!"
        pos_scores = bod_weight_pos.detach() * pos_scores
        neg_scores = bod_weight_neg.detach() * neg_scores
        if not isinstance(self.backbone, SGL):
            loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores)).mean()
        else:
            loss = -torch.log(self.bpr_gamma + torch.sigmoid(pos_scores - neg_scores)).sum()
        return loss
