import logging
import os
import pickle

import coloredlogs
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from adni.models.network import make_network
from common.adni.utils import CATEGORICAL_COLS, NUMERICAL_COLS, CATEGORY2CLASSNUM, IMAGING_COLS, remove_date_from_name
from common.losses import create_loss

MIN_NUM_PATCHES = 0

coloredlogs.install()


class CLIMATv1(nn.Module):
    def __init__(self, cfg, device, pn_weights=None, y0_weights=None):
        super().__init__()

        self.device = device
        self.cfg = cfg
        self.n_meta_out_features = 0
        self.n_all_features = 0
        self.n_last_img_features = 0
        self.n_metadata = 0
        self.n_patches = 0
        self.n_meta_features = cfg.n_meta_features
        self.input_data = cfg.parser.metadata
        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None else pn_weights

        if "IMG" in self.input_data:
            self.setup_backbone_network(cfg)
        else:
            self.n_patches = self.cfg.seq_len

        self.n_all_features += self.n_last_img_features

        for meta_name in self.input_data:
            meta_name = remove_date_from_name(meta_name)
            if meta_name in NUMERICAL_COLS:
                if meta_name in IMAGING_COLS:
                    print(f'Creating imaging feature extractor for {meta_name}.')
                    setattr(self, f'ft_{meta_name}', self.create_metadata_layers(1, self.n_last_img_features))
                    self.n_patches += 1
                else:
                    print(f'Creating feature extractor for {meta_name}.')
                    setattr(self, f'ft_{meta_name}', self.create_metadata_layers(1, self.n_meta_features))
                    self.n_meta_out_features += self.n_meta_features
                    self.n_metadata += 1

        for meta_name in self.input_data:
            if meta_name in CATEGORICAL_COLS:
                print(f'Creating feature extractor for {meta_name}.')
                setattr(self, f'ft_{meta_name}',
                        self.create_metadata_layers(CATEGORY2CLASSNUM[meta_name], self.n_meta_features))
                self.n_meta_out_features += self.n_meta_features
                self.n_metadata += 1

        self.n_all_features += self.n_meta_out_features

        self.dropout = nn.Dropout(p=cfg.drop_rate)
        self.dropout_between = nn.Dropout(cfg.drop_rate_between)

        # self.feat_patch_dim = cfg.feat_patch_dim
        self.n_classes = cfg.n_pn_classes

        self.feat_dim = cfg.feat_dim = cfg.feat_dim if isinstance(cfg.feat_dim, int) and cfg.feat_dim > 0 \
            else self.n_meta_features + self.n_last_img_features

        self.setup_transformers_with_imaging(cfg)

        self.use_tensorboard = False
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
        self.pn_weights = torch.tensor(pn_weights) if pn_weights is not None and cfg.prognosis_coef > 0 else None
        self.y0_weights = torch.tensor(y0_weights) if y0_weights is not None and cfg.diag_coef > 0 else None
        self.configure_loss_coefs(cfg)
        self.configure_crits()
        self.configure_optimizers()
        self.batch_ind = 0
        self.to(self.device)

    def setup_backbone_network(self, cfg):
        if cfg.max_depth < 1 or cfg.max_depth > 5:
            logging.fatal('Max depth must be in [1, 5].')
            assert False

        self.n_input_imgs = cfg.n_input_imgs

        self.feature_extractor = make_network(cfg, pretrained=cfg.pretrained,
                                              checkpoint_path=cfg.backbone_checkpoint_path,
                                              input_3x3=cfg.input_3x3, n_channels=cfg.n_channels)

        self.blocks = []

        if "cnn3d" in cfg.backbone_name:
            for i in range(cfg.max_depth):
                if hasattr(self.feature_extractor, f"layer{i}"):
                    self.blocks.append(getattr(self.feature_extractor, f"layer{i}"))
            # self.last_img_features = [64, 64, 128, 128, 512]
            # self.last_img_features = [64, 128, 256, 512, 1024]
            self.last_img_features = [8, 16, 32, 64, 128]
            self.sz_list = [39, 19, 13, 6, 3]
            self.n_last_img_features = self.last_img_features[cfg.max_depth - 1]
            self.n_last_img_ft_size = self.sz_list[cfg.max_depth - 1]
        elif "mobilenet" in cfg.backbone_name:
            self.blocks.append(self.feature_extractor.features)
            self.n_last_img_features = 1280
            self.n_last_img_ft_size = 3
        elif "shufflenet" in cfg.backbone_name:
            self.blocks.append(self.feature_extractor.conv1)
            self.blocks.append(self.feature_extractor.maxpool)
            self.blocks.append(self.feature_extractor.features)
            if cfg.max_depth == 5:
                self.blocks.append(self.feature_extractor.conv_last)
                self.n_last_img_ft_size = 3
                if cfg.width_mult < 2:
                    self.n_last_img_features = 1024
                elif cfg.width_mult == 2:
                    self.n_last_img_features = 2048
                else:
                    raise ValueError(f'Not support width_mult of {cfg.width_mult}.')
            elif cfg.max_depth == 4:
                self.n_last_img_ft_size = 5
                if cfg.width_mult == 1:
                    self.n_last_img_features = 464
                elif cfg.width_mult == 1.5:
                    self.n_last_img_features = 704
                elif cfg.width_mult == 2:
                    self.n_last_img_features = 976
                else:
                    raise ValueError(f'Not support width_mult of {cfg.width_mult}.')
            else:
                raise ValueError(f'Not support max_depth < 4 in shufflenet.')

        else:
            raise ValueError(f'Not support {cfg.backbone_name}.')

        self.n_img_features = cfg.n_img_features

        if self.n_img_features <= 0:
            self.img_ft_projection = nn.Identity()
        else:
            self.img_ft_projection = nn.Linear(self.n_last_img_features, cfg.n_img_features, bias=True)
            self.n_last_img_features = cfg.n_img_features

        logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

        self.n_patches += self.n_last_img_ft_size ** 3

    def setup_transformers_with_imaging(self, cfg):
        if hasattr(cfg, "num_cls_num") and cfg.num_cls_num is not None:
            self.num_cls_num = cfg.num_cls_num
        else:
            self.num_cls_num = cfg.seq_len

        self.feat_diag_dim = cfg.feat_diag_dim = cfg.feat_diag_dim if isinstance(cfg.feat_diag_dim,
                                                                                 int) and cfg.feat_diag_dim > 0 else self.n_last_img_features

        self.feat_context = FeaT(num_patches=self.n_metadata, with_cls=True, num_cls_num=1,
                                 patch_dim=self.n_meta_features,
                                 num_classes=0, dim=self.n_meta_features, depth=cfg.feat_fusion_depth,
                                 heads=cfg.feat_fusion_heads, mlp_dim=cfg.feat_fusion_mlp_dim, dropout=cfg.drop_rate,
                                 emb_dropout=cfg.feat_fusion_emb_drop_rate, n_outputs=0)

        self.feat_diagnosis = FeaT(num_patches=self.n_patches, with_cls=True, num_cls_num=1,
                                   patch_dim=self.feat_diag_dim,
                                   num_classes=cfg.n_pn_classes, dim=self.feat_diag_dim, depth=cfg.feat_diag_depth,
                                   heads=cfg.feat_diag_heads, mlp_dim=cfg.feat_diag_mlp_dim, dropout=cfg.drop_rate,
                                   emb_dropout=cfg.feat_diag_emb_drop_rate, n_outputs=cfg.feat_diag_n_outputs)

        self.feat_prognosis = FeaT(num_patches=self.n_patches + 1, with_cls=True, num_cls_num=self.num_cls_num,
                                   patch_dim=self.feat_dim,
                                   num_classes=self.n_classes, dim=self.feat_dim, depth=cfg.feat_depth,
                                   heads=cfg.feat_heads, mlp_dim=cfg.feat_mlp_dim, dropout=cfg.drop_rate,
                                   emb_dropout=cfg.feat_emb_drop_rate, n_outputs=cfg.seq_len)

    def configure_loss_coefs(self, cfg):
        self.log_vars_y0 = nn.Parameter(torch.zeros((1,))) if self.is_mtl() and self.has_y0() else torch.tensor(0.0)
        self.log_vars_pn = nn.Parameter(torch.zeros((cfg.seq_len))) if self.is_mtl() and self.has_pn() \
            else torch.zeros((cfg.seq_len), requires_grad=False)

        # alpha
        self.y0_init_power = cfg.y0_init_power
        self.pn_init_power = cfg.pn_init_power

        if cfg.diag_coef > 0:
            self.alpha_power_y0 = torch.tensor(self.y0_init_power, dtype=torch.float32)
        if cfg.prognosis_coef > 0:
            self.alpha_power_pn = torch.tensor([self.pn_init_power] * cfg.seq_len, dtype=torch.float32)

        # Show class weights
        if self.y0_weights is not None and self.y0_init_power is not None:
            _y0_weights = self.y0_weights ** self.y0_init_power
        else:
            _y0_weights = None
        if self.pn_weights is not None and self.pn_init_power is not None:
            _pn_weights = self.pn_weights ** self.pn_init_power
        else:
            _pn_weights = None
        print(f'{cfg.grading} weights:\n{_y0_weights}')
        print(f'PN weights:\n{_pn_weights}')

        # gamma
        if self.is_our_loss(with_focal=True):
            print('Force focal gamma to 1.')
            cfg.focal.gamma = 1.0
        self.gamma_y0 = float(cfg.focal.gamma) \
            if "F" in self.cfg.loss_name and self.has_y0() else None
        self.gamma_pn = np.array([float(cfg.focal.gamma)] * cfg.seq_len, dtype=float) \
            if "F" in self.cfg.loss_name and self.has_pn() else [None] * cfg.seq_len

    def configure_crits(self):
        self.crit_pn = create_loss(loss_name=self.cfg.loss_name,
                                   normalized=False,
                                   gamma=self.cfg.focal.gamma,
                                   reduction='mean').to(self.device)
        self.crit_diag = create_loss(loss_name=self.cfg.loss_name,
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

    def has_pr(self):
        return False

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

        probs = torch.softmax(x * tau, dim=dim)

        if to_numpy:
            probs = self._to_numpy(probs)
        return probs

    def forward(self, input, batch_i=None, target=None):
        meta_features = []
        img_measure_features = []
        img_features = None
        for input_type in self.input_data:
            if input_type not in input:
                print(f'Input has no {input_type}')
            elif input_type.upper() == "IMG":
                img_features = self.forward_img(input[input_type])
            else:
                _ft = getattr(self, f"ft_{input_type}")(input[input_type])
                _ft = input[f'{input_type}_mask'].view(-1, 1) * _ft
                _ft = torch.unsqueeze(_ft, 1)
                _ft = self.dropout_between(_ft)
                if input_type in IMAGING_COLS:
                    img_measure_features.append(_ft)
                else:
                    meta_features.append(_ft)

        preds, diag_preds, d_attn, f_attn, p_attn = self.apply_feat(img_features, img_measure_features, meta_features)

        # Save image, metadata names, and attention map into pkl files
        if self.cfg.save_attn:
            self.save_attentions(batch_i, root=self.cfg.log_dir, img=input['IMG'] if "IMG" in self.input_data else None,
                                 metadata_names=self.input_data, diags=diag_preds,
                                 d_attn=d_attn, f_attn=f_attn, p_attn=p_attn, preds=preds, target=target)

        return preds, diag_preds

    def save_attentions(self, batch_i, root, img, metadata_names, d_attn, f_attn, p_attn, preds, target, diags):
        diag_target = target[f'prognosis_{self.cfg.grading}'][:, 0]

        pn_target = target[f'prognosis_{self.cfg.grading}'][:, 1:]
        pn_masks = target[f'prognosis_mask_{self.cfg.grading}'][:, 1:]
        data = {'img': img.to('cpu').detach().numpy(),
                'metadata': metadata_names,
                'D': d_attn.to('cpu').detach().numpy(),
                'F': f_attn.to('cpu').detach().numpy(),
                'P': p_attn.to('cpu').detach().numpy(),
                'preds': preds.to('cpu').detach().numpy(),
                'targets': pn_target.to('cpu').detach().numpy(),
                'mask': pn_masks.to('cpu').detach().numpy(),
                'diags': diags.to('cpu').detach().numpy(),
                'y0': diag_target.to('cpu').detach().numpy()}
        os.makedirs(os.path.join(root, "attn"), exist_ok=True)

        attn_fullname = os.path.join(root, "attn", f"batch_{self.cfg.fold_index}_{batch_i:02d}.pkl")
        print(os.path.abspath(attn_fullname))
        with open(attn_fullname, 'wb') as f:
            pickle.dump(data, f, 4)

    def apply_feat(self, img_features, img_measure_features, meta_features):
        has_img = img_features is not None
        has_meta = meta_features != []

        if has_img:
            img_features = rearrange(img_features, 'b c t h w -> b (t h w) c')
            if len(img_measure_features) > 0:
                img_measure_features = torch.cat(img_measure_features, 1)
                img_features = torch.cat((img_features, img_measure_features), 1)

        diag_preds, img_descs, d_attns = self.feat_diagnosis(img_features)
        diag_preds = diag_preds.squeeze(1)

        if has_meta:
            meta_features = torch.cat(meta_features, 1)

            # Apply Fusion transformer
            _, fusion_features, f_attns = self.feat_context(meta_features)
            meta_features = fusion_features[:, 0:1, :]

            meta_features = meta_features.repeat(1, img_descs.shape[1], 1)
            meta_features = self.dropout(meta_features)
            meta_features = torch.cat((img_descs, meta_features), dim=-1)
        else:
            meta_features = img_descs
            f_attns = [None]

        preds, _, p_attns = self.feat_prognosis(meta_features)

        return preds, diag_preds, d_attns[-1], f_attns[-1], p_attns[-1]

    def create_metadata_layers(self, n_input_dim, n_output_dim):
        return nn.Sequential(
            nn.Linear(n_input_dim, n_output_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(n_output_dim)
        )

    def forward_img(self, input):
        features = []
        if isinstance(input, torch.Tensor):
            input = (input,)

        for x in input:
            for i_b, block in enumerate(self.blocks):
                if isinstance(block, list) or isinstance(block, tuple):
                    for sub_block in block:
                        x = sub_block(x)
                else:
                    x = block(x)
                if i_b < len(self.blocks) - 1:
                    x = self.dropout_between(x)
                else:
                    x = self.dropout(x)

            if self.cfg.feat_use:
                img_ft = x
            else:
                img_ft = self.gap(x)
                img_ft = img_ft.view(img_ft.shape[0], -1)
            features.append(img_ft)

        features = torch.cat(features, 1)
        return features

    def log_tau(self, batch_i, n_iters, epoch_i, stage):
        if batch_i + 1 >= n_iters and "MTL" in self.cfg.loss_name and self.use_tensorboard and stage == "eval":
            tau_y0 = {'y0': self.tau_y0} if self.has_y0() and "MTL" in self.cfg.loss_name else {}
            tau_pn = {f'pn{i}': self.tau_pn[i] for i in
                      range(self.cfg.seq_len)} if self.has_pn() and "MTL" in self.cfg.loss_name else {}
            taus = {**tau_y0, **tau_pn}
            self.writer.add_scalars('tau', taus, global_step=epoch_i)

    def apply_constraints_softmax_all(self, s=1.0):
        if "U" in self.cfg.loss_name:
            rho = []

            if self.has_y0():
                self.tau_y0 = 1.0 / (torch.exp(self.log_vars_y0) + s)
                rho.append(self.tau_y0)
            rho.append(1.0 / (torch.exp(self.log_vars_pn) + s))

            rho = torch.cat(rho, 0)
            rho_bar = torch.softmax(rho, dim=0)
            tau = rho_bar / rho_bar.max()
            if self.has_y0():
                self.tau_y0 = tau[0]
                self.tau_pn = tau[1:]
            else:
                self.tau_y0 = 1.0
                self.tau_pn = tau

        elif self.is_mtl():
            self.tau_y0 = torch.exp(-self.log_vars_y0)
            self.tau_pn = torch.exp(-self.log_vars_pn)
        else:
            self.tau_y0 = 1.0
            self.tau_pn = [1.0] * self.cfg.seq_len

    def fit(self, input, target, batch_i, n_iters, epoch_i, stage="train"):

        diag_target = target[f'prognosis_{self.cfg.grading}'][:, 0]
        diag_mask = target[f'prognosis_mask_{self.cfg.grading}'][:, 0] * input['IMG_mask']

        pn_target = target[f'prognosis_{self.cfg.grading}'][:, 1:]
        pn_masks = target[f'prognosis_mask_{self.cfg.grading}'][:, 1:]

        preds, diag_preds = self.forward(input, batch_i, target=target)

        outputs = {'pn': {'prob': [], 'label': []}}

        # Bound parameters
        self.apply_constraints_softmax_all(s=self.cfg.club.s)

        # Current KL
        cur_diag_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        if self.has_y0():
            outputs[self.cfg.grading] = {'prob': None, 'label': None}

            grading_preds_mask = diag_preds[diag_mask, :]
            grading_target_mask = diag_target[diag_mask]

            if grading_preds_mask.shape[0] > 0:
                outputs[self.cfg.grading]['prob'] = self._compute_probs(grading_preds_mask, to_numpy=True)
                outputs[self.cfg.grading]['label'] = self._to_numpy(grading_target_mask)

            if grading_target_mask.nelement() > 0:
                if self.y0_weights is None or self.alpha_power_y0 is None:
                    y0_pw_weights = None
                else:
                    self.y0_weights = self.y0_weights.to(self.alpha_power_y0.device)
                    y0_pw_weights = self.y0_weights ** self.alpha_power_y0

                cur_diag_loss = self.crit_diag(grading_preds_mask, grading_target_mask, alpha=y0_pw_weights,
                                               tau=self.tau_y0)

        if self.pn_weights is not None:
            self.pn_weights = self.pn_weights.to(self.alpha_power_pn.device)

        pn_losses = torch.zeros((self.cfg.seq_len), device=self.device)
        n_t_pn = 0
        for t in range(self.cfg.seq_len):

            pn_logits_mask = preds[pn_masks[:, t], t, :]
            pn_target_mask = pn_target[pn_masks[:, t], t]

            outputs['pn']['prob'].append(
                self._compute_probs(pn_logits_mask, self.tau_pn[t], to_numpy=True))

            outputs['pn']['label'].append(self._to_numpy(pn_target_mask))

            if self.has_pn() and pn_logits_mask.shape[0] > 0 and pn_target_mask.shape[0] > 0:
                pn_pw_weights = self.pn_weights[t, :] ** self.alpha_power_pn[t] if self.pn_weights is not None else None
                pn_loss = self.crit_pn(pn_logits_mask, pn_target_mask, normalized=False,  # log_var=self.log_vars_pn[t],
                                       tau=self.tau_pn[t],
                                       alpha=pn_pw_weights, gamma=self.gamma_pn[t])
                if torch.isnan(pn_loss):
                    print(
                        f'pn_{t} -- tau: {self.tau_pn[t].item()}, gamma: {self.gamma_pn[t].item()}, alpha: {pn_pw_weights}')
                pn_losses[t] = pn_loss
                n_t_pn += 1

        losses = {}
        prognosis_loss = torch.sum(pn_losses)
        if n_t_pn > 0:
            prognosis_loss /= n_t_pn
        else:
            prognosis_loss = torch.tensor(0.0, requires_grad=True)

        loss = self.cfg.prognosis_coef * prognosis_loss + self.cfg.diag_coef * cur_diag_loss

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
                 n_outputs=1, dropout=0., emb_dropout=0.):
        super().__init__()
        self.patch_dim = patch_dim
        self.n_outputs = n_outputs
        self.with_cls = with_cls
        if self.with_cls:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls_num, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_num, dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        for i in range(self.n_outputs):
            setattr(self, f"mlp_head{i}", nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes)
            ))

    def forward(self, features, mask=None):
        x = self.patch_to_embedding(features)

        if self.with_cls:
            cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        states, attentions = self.transformer(x, mask)

        x = self.to_cls_token(states[:, 0:self.n_outputs])

        outputs = []
        for i in range(self.n_outputs):
            out = getattr(self, f"mlp_head{i}")(x[:, i])
            outputs.append(out)

        if len(outputs) > 0:
            outputs = torch.stack(outputs, dim=1)

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
