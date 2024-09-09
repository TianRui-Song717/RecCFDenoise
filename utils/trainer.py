# -*- coding: utf-8 -*-
from time import time
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.trainer import Trainer
from recbole.utils import set_color, get_gpu_usage, early_stopping, dict2str

from model.base import GraphGenerator_VAE
from recbole_gnn.recbole_gnn.model.general_recommender import NCL, SGL


class DenoiseTrainer(Trainer):
    def __init__(self, config, model):
        super(DenoiseTrainer, self).__init__(config, model)
        # ********************** For NCL ***************
        self.is_NCL = isinstance(self.model.backbone, NCL)
        if isinstance(self.model.backbone, NCL):
            self.num_m_step = config["m_step"]
            assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # only differences from the original trainer
            if self.is_NCL:
                if epoch_idx % self.num_m_step == 0:
                    self.logger.info("Running NCL E-step!")
                    self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
                valid_step += 1
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result


class BODTrainer(DenoiseTrainer):
    def __init__(self, config, model):
        super(BODTrainer, self).__init__(config, model)
        self.inner_loop = 1                 # config["BOD_inner_loop"]
        self.outer_loop = 1                 # config["BOD_outer_loop"]
        self.weight_bpr = 1                 # config["BOD_weight_bpr"]
        self.weight_alignment = 1           # config["BOD_weight_alignment"]
        self.weight_uniformity = 0.5        # config["BOD_weight_uniformity"]

        self.generator_lr = 0.001           # config["BOD_generator_lr"]
        self.generator_reg = 1e-05          # config["BOD_generator_reg"]
        self.generator_emb_size = 64        # config["BOD_generator_emb_size"]
        self.model_generator = GraphGenerator_VAE(self.generator_emb_size).to(config["device"])
        self.generator_optimizer = torch.optim.Adam(self.model_generator.parameters(), lr=self.generator_lr)

        self.model_parameters = list(self.model.parameters())

        # ********************** For NCL ***************
        self.is_NCL = isinstance(self.model.encoder, NCL)
        if isinstance(self.model.encoder, NCL):
            self.num_m_step = config["m_step"]
            assert self.num_m_step is not None

    @staticmethod
    def uniformity_loss(x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def match_loss(gw_syn, gw_real, dis_metric):
        dis = torch.tensor(0.0).to('cuda')
        if dis_metric == 'ours':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += BODTrainer.distance_wb(gwr, gws)
        else:
            exit('DC error: unknown distance function')
        return dis

    @staticmethod
    def distance_wb(gwr, gws):
        shape = gwr.shape

        # TODO: output node!!!!
        if len(gwr.shape) == 2:
            gwr = gwr.T
            gws = gws.T
        if len(shape) == 4:  # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2:  # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return 0

        dis_weight = torch.sum(
            1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis

    @staticmethod
    def l2_reg_loss(reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()

        # TODO: OURS
        loss_func = loss_func or self.model.calculate_loss

        total_loss = 0.0 if not isinstance(self.model.encoder, NCL) else None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)


        for inner_iter in range(self.inner_loop):
            for batch_idx, interaction in enumerate(iter_data):
                interaction = interaction.to(self.device)   # pairwise
                self.optimizer.zero_grad()
                self.generator_optimizer.zero_grad()

                # Model Generator
                pos_u_idx, neg_u_idx = interaction[self.model.USER_ID], interaction[self.model.USER_ID]
                pos_i_idx, neg_i_idx = interaction[self.model.ITEM_ID], interaction[self.model.NEG_ITEM_ID]
                u_emb, i_emb = self.model.forward()
                pos_user_emb_syn, pos_item_emb_syn = u_emb[pos_u_idx], i_emb[pos_i_idx]
                neg_user_emb_syn, neg_item_emb_syn = u_emb[neg_u_idx], i_emb[neg_i_idx]
                A_weight_user_item_full = self.model_generator(pos_user_emb_syn, pos_item_emb_syn)
                A_weight_user_item_full_neg = self.model_generator(neg_user_emb_syn, neg_item_emb_syn)
                A_weight_inner_full_detach = A_weight_user_item_full.detach()
                A_weight_inner_full_neg_detach = A_weight_user_item_full_neg.detach()

                # # TODO ORIGIN: convert to model.calculate_loss
                # # weighted BPR
                # pos_score = A_weight_inner_full_detach * torch.mul(pos_user_emb_syn, pos_item_emb_syn).sum(dim=1)
                # neg_score = A_weight_inner_full_neg_detach * torch.mul(pos_user_emb_syn, neg_item_emb_syn).sum(dim=1)
                # loss = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score))
                # bpr_inner = torch.mean(loss) # Rec Loss
                # # Other Loss
                # cl_loss = 0.05 * 0.0 # TODO WHY: 0.005 calculate Other Losses (e.g. SGL & NCL) ???

                # TODO: OURS
                losses = loss_func(
                    interaction, bod_weight_pos=A_weight_inner_full_detach, bod_weight_neg=A_weight_inner_full_neg_detach
                )

                # alignment_loss_weight
                x, y = F.normalize(pos_user_emb_syn, dim=-1), F.normalize(pos_item_emb_syn, dim=-1)
                align_loss = (x - y).norm(p=2, dim=1).pow(2)
                alignment_inner = (A_weight_inner_full_detach * align_loss).mean()

                # uniformity_inner
                uniformity_inner = (self.uniformity_loss(pos_user_emb_syn) + self.uniformity_loss(neg_user_emb_syn)
                                    + self.uniformity_loss(pos_item_emb_syn) + self.uniformity_loss(
                            neg_item_emb_syn)) / 4

                # TODO ORIGIN
                # batch_loss_inner = self.weight_bpr * bpr_inner + self.weight_alignment * alignment_inner + self.weight_uniformity * uniformity_inner + cl_loss

                # TODO: OURS
                if isinstance(losses, tuple):
                    rec_loss = sum(losses)
                else:
                    rec_loss = losses
                batch_loss_inner = rec_loss + self.weight_alignment * alignment_inner + self.weight_uniformity * uniformity_inner

                # OURS: NCL Loss Tuple
                if self.is_NCL:
                    if isinstance(losses, tuple):
                        if epoch_idx < self.config['warm_up_step']:
                            losses = losses[:-1]
                        rec_loss = sum(losses)
                        loss_tuple = tuple(per_loss.item() for per_loss in losses)
                        total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                    else:
                        rec_loss = losses
                        total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                else:
                    if isinstance(losses, tuple):
                        rec_loss = sum(losses)
                        loss_tuple = tuple(per_loss.item() for per_loss in losses)
                        total_loss = (
                            loss_tuple
                            if total_loss is None
                            else tuple(map(sum, zip(total_loss, loss_tuple)))
                        )
                    else:
                        rec_loss = losses
                        total_loss = (
                            losses.item() if total_loss is None else total_loss + losses.item()
                        )
                self._check_nan(rec_loss)
                batch_loss_inner.backward(retain_graph=True)
                self.optimizer.step()

                if self.gpu_available and show_progress:
                    iter_data.set_postfix_str(
                        set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                    )

        self.model_generator.train()
        ol_iter_data = iter(train_data)
        for ol_iter in range(self.outer_loop):
            self.generator_optimizer.zero_grad()
            loss = torch.tensor(0.0).to('cuda')
            interaction = next(ol_iter_data)
            interaction = interaction.to(self.device)  # pairwise

            # Model Generator
            u_idx_ol = interaction[self.model.USER_ID]
            i_idx_ol, j_idx_ol = interaction[self.model.ITEM_ID], interaction[self.model.NEG_ITEM_ID]
            pos_u_idx_ol = u_idx_ol
            pos_i_idx_ol = i_idx_ol
            neg_i_idx_ol = j_idx_ol

            u_emb, i_emb = self.model.forward()
            user_emb_ol, item_emb_ol  = u_emb[u_idx_ol], i_emb[i_idx_ol]
            pos_user_emb_ol, pos_item_emb_ol, neg_item_emb_ol = u_emb[pos_u_idx_ol], i_emb[pos_i_idx_ol], i_emb[neg_i_idx_ol]

            A_weight_user_item_pos = self.model_generator(pos_user_emb_ol, pos_user_emb_ol)
            A_weight_user_item_neg = self.model_generator(pos_user_emb_ol, neg_item_emb_ol)

            # weighted BPR
            pos_score = A_weight_user_item_pos * torch.mul(pos_user_emb_ol, pos_item_emb_ol).sum(dim=1)
            neg_score = A_weight_user_item_neg * torch.mul(pos_user_emb_ol, neg_item_emb_ol).sum(dim=1)
            bpr_loss_ol = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).mean()
            gw_real = torch.autograd.grad(bpr_loss_ol, self.model_parameters, retain_graph=True, create_graph=True)

            A_weight_user_item = self.model_generator(user_emb_ol, item_emb_ol)
            # alignment_loss_weight
            x, y = F.normalize(user_emb_ol, dim=-1), F.normalize(item_emb_ol, dim=-1)
            loss = (x - y).norm(p=2, dim=1).pow(2)
            alignment_syn_ol = (A_weight_user_item * loss).mean()

            gw_syn = torch.autograd.grad(alignment_syn_ol, self.model_parameters, retain_graph=True, create_graph=True)
            loss = self.match_loss(gw_real, gw_syn, 'ours')
            loss_reg = self.l2_reg_loss(self.generator_reg, user_emb_ol, item_emb_ol)
            loss = loss + loss_reg

            loss.backward()
            self.generator_optimizer.step()
        self.model_generator.eval()

        return total_loss
