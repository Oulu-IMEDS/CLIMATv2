import logging
import os
import pickle

import coloredlogs
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from common.losses import create_loss
from prognosis.models.networks import get_output_channels, make_global_pool

MIN_NUM_PATCHES = 0

coloredlogs.install()


class CLIMATv2(nn.Module):
    def __init__(self, cfg, device, pn_weights=None, y0_weights=None):
        super().__init__()
        # 'Img'
        # 'Age', 'BMI',	'KL',
        # 'Gender', 'Varus', 'Tenderness',
        # 'Injury_history', 'Mild_symptoms', 'Heberden',
        # 'Crepitus', 'Morning_stiffness', 'Postmeno'
        self.device = device
        self.cfg = cfg
        self.seq_len = cfg.seq_len + 1
        self.n_meta_out_features = 0
        self.n_meta_features = cfg.n_meta_features
        self.input_data = cfg.parser.metadata
        self.configure_output_heads_positions(cfg)

        if "AGE" in self.input_data:
            self.age_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "WOMAC" in self.input_data:
            self.womac_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "BMI" in self.input_data:
            self.bmi_ft = self.create_metadata_layers(4, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "SEX" in self.input_data:
            self.sex_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "INJ" in self.input_data:
            self.inj_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "SURG" in self.input_data:
            self.surg_ft = self.create_metadata_layers(2, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "KL" in self.input_data:
            self.kl_ft = self.create_metadata_layers(cfg.n_pn_classes, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "META_M" in self.input_data:
            self.meta_m_ft = self.create_metadata_layers(11, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        if "META_D" in self.input_data:
            self.meta_d_ft = self.create_metadata_layers(11, self.n_meta_features)
            self.n_meta_out_features += self.n_meta_features

        self.n_all_features = self.n_meta_out_features
        if "IMG" in self.input_data:
            if cfg.max_depth < 1 or cfg.max_depth > 5:
                logging.fatal('Max depth must be in [1, 5].')
                assert False

            self.n_input_imgs = cfg.n_input_imgs

            if cfg.use_bn:
                from common.models.networks_bn import make_network
            else:
                from common.models.networks_in import make_network
            self.feature_extractor = make_network(name=cfg.backbone_name, pretrained=cfg.pretrained,
                                                  input_3x3=cfg.input_3x3)

            self.blocks = []
            for i in range(cfg.max_depth):
                if hasattr(self.feature_extractor, f"layer{i}"):
                    self.blocks.append(getattr(self.feature_extractor, f"layer{i}"))
                elif i == 0 and 'resnet' in cfg.backbone_name:
                    self.blocks.append([self.feature_extractor.conv1, self.feature_extractor.bn1,
                                        self.feature_extractor.relu, self.feature_extractor.maxpool])

            if self.cfg.dataset == "toy":
                self.sz_list = [32, 16, 8]  # 32x32
                self.n_last_img_features = 64
                self.n_last_img_ft_size = 8
            elif self.cfg.dataset == "mnist3x3":
                self.sz_list = [64, 32, 16, 8, 4]  # 128x128
                self.n_last_img_features = get_output_channels(self.blocks[-1], cfg.max_depth)
                self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]
            else:
                # self.sz_list = [75, 75, 38, 19, 10] # 300x300
                self.sz_list = [64, 64, 32, 16, 8]  # 256x256
                self.n_last_img_features = get_output_channels(self.blocks[-1], cfg.max_depth)
                self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]

            self.n_img_features = cfg.n_img_features

            if cfg.global_pool_name:
                self.pool = make_global_pool(pool_name=cfg.global_pool_name, channels=self.n_img_features,
                                             use_bn=cfg.use_bn, sz=self.sz_list[cfg.max_depth - 1])

                use_sam = "sam" in cfg.global_pool_name
                self.n_last_img_ft_size = 1
                if use_sam:
                    self.n_last_img_features = self.pool.output_channels  # self._feature_channels * 2

            if self.n_img_features <= 0:
                self.img_ft_projection = nn.Identity()
            else:
                self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
                self.n_last_img_features = cfg.n_img_features

            self.n_all_features += self.n_last_img_features
            logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

            # self.meta_ft_adjustment = nn.Parameter(torch.ones(1, 1, self.n_meta_out_features))
            self.n_patches = self.n_last_img_ft_size * self.n_last_img_ft_size
        else:
            # self.meta_ft_adjustment = nn.Parameter(torch.ones(1, 1, self.n_all_features))
            self.n_patches = self.seq_len

        self.dropout = nn.Dropout(p=cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

        # self.feat_patch_dim = cfg.feat_patch_dim
        self.n_classes = cfg.n_pn_classes

        self.feat_dim = cfg.feat_dim = cfg.feat_dim if isinstance(cfg.feat_dim, int) and cfg.feat_dim > 0 \
            else self.n_meta_features + self.n_last_img_features

        if hasattr(cfg, "num_cls_num"):
            self.num_cls_num = cfg.num_cls_num
        else:
            self.num_cls_num = self.seq_len

        self.feat_kl_dim = cfg.feat_kl_dim = cfg.feat_kl_dim if isinstance(cfg.feat_kl_dim,
                                                                           int) and cfg.feat_kl_dim > 0 else self.n_last_img_features

        # Fusion
        self.n_metadata = len(self.input_data) - 1  # Remove current KL (y_0)
        self.feat_fusion = FeaT(num_patches=self.n_metadata, with_cls=True, num_cls_num=1,
                                patch_dim=self.n_meta_features, gradings=(),
                                num_classes=0, dim=self.n_meta_features, depth=cfg.feat_fusion_depth,
                                heads=cfg.feat_heads, mlp_dim=cfg.feat_fusion_mlp_dim, dropout=cfg.drop_rate,
                                emb_dropout=cfg.feat_fusion_emb_drop_rate, n_outputs_per_grading=0)

        self.feat_diag = FeaT(num_patches=self.n_patches, with_cls=True, num_cls_num=1,
                              patch_dim=self.feat_kl_dim, gradings=self.gradings,
                              use_nonlinear_dec=cfg.use_nonlinear_dec,
                              num_classes=cfg.n_pn_classes, dim=self.feat_kl_dim, depth=cfg.feat_kl_depth,
                              heads=cfg.feat_heads, mlp_dim=cfg.feat_kl_mlp_dim, dropout=cfg.drop_rate,
                              emb_dropout=cfg.feat_kl_emb_drop_rate, n_outputs_per_grading=1)

        self.feat_prognosis = FeaT(num_patches=self.n_patches + 1, with_cls=True, num_cls_num=self.num_cls_num,
                                   patch_dim=self.n_meta_features + self.n_last_img_features,
                                   gradings=self.gradings, use_nonlinear_dec=cfg.use_nonlinear_dec,
                                   num_classes=cfg.n_pn_classes, dim=self.feat_dim, depth=cfg.feat_depth,
                                   heads=cfg.feat_heads, mlp_dim=cfg.feat_mlp_dim, dropout=cfg.drop_rate,
                                   emb_dropout=cfg.feat_emb_drop_rate, n_outputs_per_grading=self.seq_len,
                                   use_separate_ffn=cfg.use_separate_ffn)

        self.output_map = self.feat_prognosis.output_map

        self.use_tensorboard = False
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
        self.y0_weights = torch.tensor(y0_weights) if y0_weights is not None and cfg.diag_coef > 0 else None
        if pn_weights is not None and cfg.prognosis_coef > 0:
            if self.y0_weights is None:
                self.pn_weights = torch.tensor(pn_weights)
            else:
                self.pn_weights = torch.cat((self.y0_weights.unsqueeze(0), torch.tensor(pn_weights)), 0)
        else:
            self.pn_weights = None
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.configure_optimizers()
        self.batch_ind = 0
        self.to(self.device)

    def configure_loss_coefs(self, cfg):
        self.log_vars_pn = nn.Parameter(torch.zeros((self.seq_len))) if self.is_mtl() and self.has_pn() \
            else torch.zeros((self.seq_len), requires_grad=False)

        # alpha
        self.pn_init_power = cfg.pn_init_power

        if cfg.prognosis_coef > 0:
            self.alpha_power_pn = torch.tensor([self.pn_init_power] * self.seq_len, dtype=torch.float32)

        # Show class weights
        if self.pn_weights is not None and self.pn_init_power is not None:
            _pn_weights = self.pn_weights ** self.pn_init_power
        else:
            _pn_weights = None

        print(f'PN weights:\n{_pn_weights}')

        # gamma
        if self.is_our_loss(with_focal=True):
            print('Force focal gamma to 1.')
            cfg.focal.gamma = 1.0
        self.gamma_pn = np.array([float(cfg.focal.gamma)] * self.seq_len, dtype=float) \
            if "F" in self.cfg.loss_name and self.has_pn() else [None] * self.seq_len

    def configure_crits(self):
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   gamma=self.cfg.focal.gamma,
                                   reduction='mean').to(self.device)

    def get_params(self):
        self.params_main = []
        self.params_extra = []
        for p in self.named_parameters():
            if self.is_log_vars(p[0]) and self.is_mtl():
                self.params_extra.append(p[1])
            else:
                self.params_main.append(p[1])
        # self.params_extra = []

    def configure_optimizers(self):
        self.get_params()

        self.optimizer = torch.optim.Adam(self.params_main, lr=self.cfg['lr'],
                                          betas=(self.cfg['beta1'], self.cfg['beta2']))

        if self.is_mtl() and self.params_extra:
            self.optimizer_extra = torch.optim.Adam(self.params_extra, lr=self.cfg.extra_optimizer.lr,
                                                    betas=(
                                                        self.cfg.extra_optimizer.beta1, self.cfg.extra_optimizer.beta2))

    def is_log_vars(self, x):
        return "log_vars" in x

    def is_gamma(self, x):
        return "gamma" in x

    def has_y0(self):
        return self.cfg.diag_coef > 0

    def has_pn(self):
        return self.cfg.prognosis_coef > 0

    def is_focal_loss(self):
        return "F" in self.cfg.loss_name

    def is_upper_loss(self):
        return "U" in self.cfg.loss_name

    def is_mtl(self):
        return "MTL" in self.cfg.loss_name

    def is_our_loss(self, with_focal=False):
        if not with_focal:
            return "FMTL" in self.cfg.loss_name or "UMTL" in self.cfg.loss_name
        else:
            return "FMTL" in self.cfg.loss_name

    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy()

    def _compute_probs(self, x, tau=1.0, dim=-1, to_numpy=True):
        tau = tau if tau is not None else 1.0

        if self.cfg.loss_name == 'UMTL':
            probs = tau * torch.softmax(x, dim=dim)
        else:
            probs = torch.softmax(x * tau, dim=dim)

        if to_numpy:
            probs = self._to_numpy(probs)
        return probs

    def forward(self, input, stage, batch_i=None, target=None):
        meta_features = []
        img_features = None
        for input_type in self.input_data:
            if input_type.lower() == "img":
                img_features = self.forward_img(input[input_type.upper()])
            else:
                _ft = getattr(self, f"{input_type.lower()}_ft")(input[input_type.upper()])
                _ft = input[f'{input_type}_mask'].view(-1, 1) * _ft
                _ft = torch.unsqueeze(_ft, 1)
                # _ft = self.dropout_between(_ft)
                meta_features.append(_ft)

        preds_r, preds_p, d_attn, f_attn, p_attn = self.apply_feat(img_features, meta_features, stage)

        # Save image, metadata names, and attention map into pkl files
        if self.cfg.save_attn:
            self.save_attentions(batch_i, root=self.cfg.log_dir, img=input['IMG'] if "IMG" in self.input_data else None,
                                 metadata_names=self.input_data, diags=preds_r[0],
                                 d_attn=d_attn, f_attn=f_attn, p_attn=p_attn, preds=preds_p[1:], targets=target)

        return preds_r, preds_p

    def save_attentions(self, batch_i, root, img, metadata_names, d_attn, f_attn, p_attn, preds, targets, diags):
        data = {'img': img.to('cpu').detach().numpy(),
                'metadata': metadata_names,
                'D': d_attn.to('cpu').detach().numpy(),
                'F': f_attn.to('cpu').detach().numpy(),
                'P': p_attn.to('cpu').detach().numpy(),
                'preds': torch.stack(preds, dim=1).to('cpu').detach().numpy(),
                'targets': targets[f'prognosis_{self.gradings[0]}'][:, 1:].to('cpu').detach().numpy(),
                'mask': targets[f'prognosis_mask_{self.gradings[0]}'][:, 1:].to('cpu').detach().numpy(),
                'diags': diags.to('cpu').detach().numpy(),
                'y0': targets[f'prognosis_{self.gradings[0]}'][:, 0].to('cpu').detach().numpy()}

        os.makedirs(os.path.join(root, "attn"), exist_ok=True)
        attn_fullname = os.path.join(root, "attn", f"batch_{self.cfg.site}_{batch_i:02d}_{self.gradings[0]}.pkl")
        print(os.path.abspath(attn_fullname))
        with open(attn_fullname, 'wb') as f:
            pickle.dump(data, f, 4)

    def apply_feat(self, img_features, meta_features, stage):
        has_img = img_features is not None
        has_meta = meta_features != []

        if has_img:
            img_features = rearrange(img_features, 'b c h w -> b (h w) c')

        preds_r, img_descs, d_attns = self.feat_diag(img_features)
        # kl_preds = kl_preds.squeeze(1)

        if has_meta:
            # meta_features.append(kl_features)
            meta_features = torch.cat(meta_features, 1)

            # Apply Fusion transformer
            _, fusion_features, f_attns = self.feat_fusion(meta_features)
            meta_features = fusion_features[:, 0:1, :]
            # meta_features = self.meta_ft_adjustment * meta_features

            meta_features = meta_features.repeat(1, img_descs.shape[1], 1)
            # meta_features = self.dropout(meta_features)
            meta_features = torch.cat((img_descs, meta_features), dim=-1)
        else:
            # meta_features = kl_features
            meta_features = img_descs
            f_attns = [None]

        preds_p, _, p_attns = self.feat_prognosis(meta_features)

        return preds_r, preds_p, d_attns[-1], f_attns[-1], p_attns[-1]  # , (diag_preds[:, 0, :], preds[:, 0, :])

    def create_metadata_layers(self, n_input_dim, n_output_dim):
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
            # nn.Linear(n_input_dim, self.n_meta_features, bias=True),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.n_meta_features)
        )

    def forward_img(self, input):
        features = []
        if isinstance(input, torch.Tensor):
            input = (input,)

        for x in input:
            for block in self.blocks:
                if isinstance(block, list) or isinstance(block, tuple):
                    for sub_block in block:
                        x = sub_block(x)
                else:
                    x = block(x)
                x = self.dropout_between(x)
            if self.cfg.global_pool_name:
                x = self.pool(x)
            # x = x.permute(0, 2, 3, 1)
            # img_ft = self.img_ft_projection(x)
            # img_ft = self.dropout(img_ft)
            # img_ft = rearrange(img_ft, 'b h w c-> b (h w) c')
            x = x.permute(0, 2, 3, 1)
            img_ft = self.img_ft_projection(x)
            # img_ft = self.dropout(img_ft)
            # img_ft = rearrange(img_ft, 'b h w c-> b (h w) c')
            img_ft = img_ft.permute(0, 3, 1, 2)

            features.append(img_ft)

        features = torch.cat(features, 1)
        return features

    def log_tau(self, batch_i, n_iters, epoch_i, stage):
        if batch_i + 1 >= n_iters and "MTL" in self.cfg.loss_name and self.use_tensorboard and stage == "eval":
            tau_y0 = {'y0': self.tau_y0} if self.has_y0() and "MTL" in self.cfg.loss_name else {}
            tau_pn = {f'pn{i}': self.tau_pn[i] for i in
                      range(self.seq_len)} if self.has_pn() and "MTL" in self.cfg.loss_name else {}
            taus = {**tau_y0, **tau_pn}
            self.writer.add_scalars('tau', taus, global_step=epoch_i)

    def apply_constraints_softmax_all(self, s=1.0):
        if "U" in self.cfg.loss_name:
            _logits = []

            _logits.append(1.0 / (torch.exp(self.log_vars_pn) + s))

            _logits = torch.cat(_logits, 0)
            _softmax = torch.softmax(_logits, dim=0)
            self.tau_pn = _softmax / _softmax.max()
        elif self.is_mtl():
            self.tau_pn = torch.exp(-self.log_vars_pn)
        else:
            self.tau_pn = [1.0] * self.seq_len

    def configure_output_heads_positions(self, cfg):
        self.gradings = cfg.parser.output
        self.grading_to_output_heads = {}
        for id, grading in enumerate(self.gradings):
            self.grading_to_output_heads[grading] = [i for i in
                                                     range(id * (1 + cfg.seq_len), (id + 1) * (1 + cfg.seq_len))]

    def fit(self, input, target, batch_i, n_iters, epoch_i, stage="train"):
        preds_r, preds_p = self.forward(input, batch_i=batch_i, target=target, stage=stage)

        outputs = {}

        # Bound parameters
        self.apply_constraints_softmax_all(s=self.cfg.club.s)

        if self.pn_weights is not None:
            self.pn_weights = self.pn_weights.to(self.alpha_power_pn.device)

        cons_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        for g_i, grading in enumerate(self.gradings):
            head_index = self.output_map[f'{grading}_0'][1]
            cons_loss = cons_loss + torch.sum(F.l1_loss(preds_r[g_i], preds_p[head_index], reduction='none')) / (
                torch.numel(preds_r[g_i]))
        cons_loss = cons_loss / len(self.gradings)

        cur_diag_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
        pn_losses = torch.zeros((self.seq_len * len(self.gradings)), device=self.device)
        n_t_pn = 0
        for grading in self.gradings:
            outputs[grading] = {'prob': None, 'label': None}
            outputs[f'{grading}:pn'] = {'prob': [], 'label': []}
            for t in range(0, self.seq_len):
                head_index = self.output_map[f'{grading}_{t}'][1]
                pn_targets = target[f'prognosis_{grading}']
                pn_masks = target[f'prognosis_mask_{grading}']
                pn_logits_mask = preds_p[head_index][pn_masks[:, t], :]
                pn_target_mask = pn_targets[pn_masks[:, t], t].to(self.device)

                if t > 0:
                    outputs[f'{grading}:pn']['prob'].append(
                        self._compute_probs(pn_logits_mask, self.tau_pn[t], to_numpy=True))
                    outputs[f'{grading}:pn']['label'].append(self._to_numpy(pn_target_mask))
                else:
                    outputs[grading]['prob'] = self._compute_probs(pn_logits_mask, self.tau_pn[t], to_numpy=True)
                    outputs[grading]['label'] = self._to_numpy(pn_target_mask).flatten()

                if pn_logits_mask.shape[0] > 0 and pn_target_mask.shape[0] > 0:
                    pn_pw_weights = self.pn_weights[t, :] ** self.alpha_power_pn[
                        t] if self.pn_weights is not None else None
                    pn_loss = self.crit_pn(pn_logits_mask, pn_target_mask, normalized=False,
                                           # log_var=self.log_vars_pn[t],
                                           tau=self.tau_pn[t],
                                           alpha=pn_pw_weights, gamma=self.gamma_pn[t])
                    if torch.isnan(pn_loss):
                        print(
                            f'pn_{t} -- tau: {self.tau_pn[t].item()}, gamma: {self.gamma_pn[t].item()}, alpha: {pn_pw_weights}')

                    if t == 0:
                        cur_diag_loss = cur_diag_loss + pn_loss
                    else:
                        pn_losses[head_index] = pn_loss
                        n_t_pn += 1

        losses = {}
        cur_diag_loss = cur_diag_loss / len(self.gradings)
        prognosis_loss = torch.sum(pn_losses)
        if n_t_pn > 0:
            prognosis_loss /= n_t_pn
        else:
            prognosis_loss = torch.tensor(0.0, requires_grad=True)

        loss = self.cfg.prognosis_coef * prognosis_loss + self.cfg.diag_coef * cur_diag_loss + self.cfg.cons_coef * cons_loss

        losses['loss_y0'] = cur_diag_loss.item()
        losses['loss_pn'] = prognosis_loss.item()
        losses['loss'] = loss.item()

        if stage == "train":
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.params_main, self.cfg.clip_norm)
                self.optimizer.step()

        if self.is_our_loss() and torch.isnan(loss):
            print(f'loss_y0: {cur_diag_loss}, loss_pn: {prognosis_loss}.')

        if stage == "eval" and self.is_mtl() and not torch.isnan(loss):
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                if self.cfg.clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.params_extra, self.cfg.clip_norm)
                if batch_i + 1 >= n_iters:
                    self.optimizer_extra.step()
                    self.optimizer_extra.zero_grad()

        return losses, outputs


class FeaT(nn.Module):
    def __init__(self, num_patches, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_cls_num=1, with_cls=True,
                 n_outputs_per_grading=1, dropout=0., emb_dropout=0., use_separate_ffn=False, gradings=(),
                 use_nonlinear_dec=True):
        super().__init__()
        self.patch_dim = patch_dim
        self.n_outputs_per_grading = n_outputs_per_grading
        self.gradings = gradings
        self.with_cls = with_cls
        self.use_separate_ffn = use_separate_ffn
        if self.with_cls:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls_num, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_num, dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.use_nonlinear_dec = use_nonlinear_dec
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.output_map = {}
        if self.use_separate_ffn:
            for g_i, grading in enumerate(gradings):
                if grading == 'KL':
                    num_classes = 5
                elif grading in ['JSL', 'JSM', 'OSFL', 'OSFM', 'OSTL', 'OSTM']:
                    num_classes = 4
                else:
                    raise ValueError(f'Not support grading `{grading}`')

                for i in range(self.n_outputs_per_grading):
                    head_name = f"mlp_head_{grading}_{i}"
                    if self.use_nonlinear_dec:
                        setattr(self, head_name, nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim // 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim // 2, num_classes)
                        ))
                    else:
                        setattr(self, head_name, nn.Linear(dim, num_classes))

                    self.output_map[f"{grading}_{i}"] = (head_name, g_i * self.n_outputs_per_grading + i)
        else:
            for g_i, grading in enumerate(gradings):
                if grading == 'KL':
                    num_classes = 5
                    if len(gradings) == 1:
                        head_name = 'mlp_head'
                    else:
                        head_name = f'mlp_head_{grading}'
                elif grading in ['JSL', 'JSM', 'OSFL', 'OSFM', 'OSTL', 'OSTM']:
                    num_classes = 4
                    head_name = f'mlp_head_{grading}'
                else:
                    raise ValueError(f'Not support grading `{grading}`')

                if self.use_nonlinear_dec:
                    setattr(self, head_name, nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim // 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim // 2, num_classes)
                    ))
                else:
                    setattr(self, head_name, nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes)))

                for i in range(self.n_outputs_per_grading):
                    self.output_map[f"{grading}_{i}"] = (head_name, g_i * self.n_outputs_per_grading + i)

    def forward(self, features, mask=None):
        x = self.patch_to_embedding(features)

        if self.with_cls:
            cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        states, attentions = self.transformer(x, mask)

        x = self.to_cls_token(states)

        x = self.dropout(x)

        outputs = []

        for grading in self.gradings:
            for i in range(self.n_outputs_per_grading):
                head_name = self.output_map[f"{grading}_{i}"][0]
                head_index = self.output_map[f"{grading}_{i}"][1]
                out = getattr(self, head_name)(x[:, head_index])
                outputs.append(out)

        # if len(outputs) > 0:
        #     outputs = torch.stack(outputs, dim=1)

        return outputs, states, attentions


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.depth = depth

        for d in range(depth):
            setattr(self, f"prenorm_0_{d}", nn.LayerNorm(dim))
            setattr(self, f"attn_{d}", Attention(dim, heads=heads, dropout=dropout))

            setattr(self, f"prenorm_1_{d}", nn.LayerNorm(dim))
            setattr(self, f"ff_{d}", FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x, mask=None):
        attentions = []
        for d in range(self.depth):
            o = getattr(self, f"prenorm_0_{d}")(x)
            o, attn = getattr(self, f"attn_{d}")(o, mask)
            attentions.append(attn)
            x = o + x

            ff = getattr(self, f"prenorm_1_{d}")(x)
            ff = getattr(self, f"ff_{d}")(ff)
            x = ff + x

        return x, attentions
