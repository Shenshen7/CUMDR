import torch
from models import CUMDR, load_pretrained, AllGather
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .uncertainty_aware import gaussian_modeling
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import viridis
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import norm
import random
class CUMDR_Retrieval(CUMDR):
    def __init__(self, config):
        super().__init__(config, load_vision_params=config['load_params'], load_text_params=config['load_params'],
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'])
        if not self.pa100k_only_img_classifier:
            self.mlm = config['mlm']
            self.pa100k = config['pa100k']
            self.total_epoch = config['schedular']['epochs']

            self.lu = config['lu']
            self.unc = config['unc']
            self.mim = config['mim']
            self.sdm = config['sdm']
            self.id = config['id']
            self.match = config['match']
            self.diffusion = config['diffusion']
            self.uncertainty = config['uncertainty']
            self.num_diffusion_query = config['num_diffusion_query']
            self.fine_grained = config['fine_grained']
            self.gpt = config['gpt']
            if not self.pa100k:
                self.eda = config['eda']
            if ('attr' in config.keys()) and config['attr']:
                self.attr = True
                self.t = config['t']
            else:
                self.attr = False

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None, 
                cur_epoch=None, diffusion=None, schedule_sampler=None, gpt_input=None):
        
        add_loss = dict()
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                            text_embeds, text_atts, text_feat, idx=idx)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                        masked_ids)
        add_loss.update({'loss_mlm': loss_mlm})
        if self.attr:

            attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
            attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

            attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
            attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, attr_text_embeds, attr_text_atts,
                                                        label)


            attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds, image_atts,
                                                    attr_masked_pos, attr_masked_ids, label)
            loss_attr = self.t * (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3
            add_loss.update({'loss_attr': loss_attr})

        
        # eda
        if self.eda:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda
        add_loss.update({'loss_itc': loss_itc})
        add_loss.update({'loss_itm': loss_itm})

        if self.lu:
            augmented_features = self.add_gaussian_noisy(image_feat)
            loss_lu = self.get_lu_loss(image_feat, text_feat, augmented_features, self.total_epoch, cur_epoch)
            add_loss.update({'loss_lu': loss_lu})
        if self.unc:
            loss_unc = self.get_unc_loss(image_feat, text_feat, idx-1, cur_epoch, self.total_epoch)
            add_loss.update({'loss_unc': loss_unc})
        if self.mim:
            mask_for_one_batch = self.build_masks_for_one_batch(image.shape[0])
            masked_img = self.build_masked_image(image, mask_for_one_batch)
            masked_img_embed, masked_img_atts = self.get_vision_embeds(masked_img)
            masked_img_feat = self.get_features(masked_img_embed, None)
            recon_image = self.mim_gen(masked_img_feat, text_feat)
            # recon_image = recon_image.reshape([masked_img_feat.shape[0], 3, 384, 128])
            temp_image = self.get_unmasked_image(image, mask_for_one_batch).reshape([image.shape[0], 3*384*96])
            loss_mim = self.get_mim_loss(recon_image, temp_image)
            add_loss.update({'loss_mim': loss_mim})
        if self.sdm:
            loss_sdm = self.get_sdm_loss(image_feat, text_feat, logit_scale=self.temp, pid=idx-1)
            add_loss.update({'loss_sdm': loss_sdm})
        if self.id:
            image_logits = self.classfier(image_feat)
            text_logits = self.classfier(text_feat)
            loss_id = self.get_id_loss(image_logits, text_logits, idx-1)
            add_loss.update({'loss_id': loss_id})
        if self.match:
            loss_match = self.get_match_loss(image_feat, text_feat)
            add_loss.update({'loss_match': loss_match})
        if self.diffusion: #diffusion

            pos = torch.ones((image_feat.shape[0], 1), dtype=torch.float)
            neg = torch.zeros((image_feat.shape[0], self.num_diffusion_query), dtype=torch.float)
            micro = torch.cat([pos, neg], dim=1).to(image_feat.device)

            t, _ =schedule_sampler.sample(image_feat.shape[0], text_feat.device)

            diffusion_img_feat = image_feat.detach()
            diffusion_text_feat = text_feat.detach()
            t2i_sim_logits = diffusion_text_feat @ diffusion_img_feat.t()
            t2i_sim_logits = F.softmax(t2i_sim_logits, dim=1)
            mask = torch.eq(idx.unsqueeze(0), idx.unsqueeze(0).t())
            t2i_sim_logits.masked_fill_(mask, -1)
            image_feat_neg = []
            for b in range(t2i_sim_logits.size(0)):
                _, neg_idx = t2i_sim_logits[b].topk(self.num_diffusion_query, largest=True, sorted=True)
                temp = [diffusion_img_feat[b]]
                for i in neg_idx:
                    temp.append(diffusion_img_feat[i])
                image_feat_neg.append(torch.stack(temp, dim=0))
            image_feat_neg = torch.stack(image_feat_neg, dim=0)
            # print(image_feat_neg.shape)
            output = diffusion.training_losses(self.diffusion_model, 
                                               micro, 
                                               t, 
                                               {"text_feat": diffusion_text_feat,
                                                "image_feat": image_feat_neg},
                                                temp = 1)
            loss_diffusion = output["kl_loss"]
            add_loss.update({'loss_diffusion': loss_diffusion})
        if self.uncertainty:
            uncertain_img_feat = self.uncertain_image_proj(image_embeds)
            uncertain_txt_feat = self.uncertain_text_proj(text_embeds)
            contrast_uncertain_img_mu, contrast_uncertain_img_logsigma, _ = self.img_gaussian(uncertain_img_feat, image_atts)
            contrast_uncertain_txt_mu, contrast_uncertain_txt_logsigma, _ = self.text_gaussian(uncertain_txt_feat, text_atts)
            # D-VLC (contrast)
            ldvlc = self.get_dvlc_loss(contrast_uncertain_img_logsigma, contrast_uncertain_txt_logsigma, 
                                      contrast_uncertain_img_mu, contrast_uncertain_txt_mu)
            # d-mlm
            mlm_embeds = self.get_mlm_embeds(text_ids_masked, text_atts, image_embeds, image_atts)
            # mlm_embeds = self.uncertain_text_proj(mlm_embeds)
            _, mlm_txt_feat, _, _, _, mlm_txt_logsigma = gaussian_modeling(
                image_embeds=None, extend_image_masks=None, 
                text_embeds=mlm_embeds, extend_text_masks=text_atts, 
                img_gau_encoder=None, txt_gau_encoder=self.text_mlm_gaussian,mu_num=1, sample_num=5)
            lmlm = self.get_dmlm_loss(mlm_txt_feat, masked_ids, contrast_uncertain_img_logsigma, mlm_txt_logsigma, 5, masked_pos)

            # d-itm
            lditm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                    text_embeds, text_atts, text_feat, idx=idx, uncertainty=True)

            loss_irtr = ldvlc + lmlm + lditm
            add_loss.update({'loss_irtr': loss_irtr})
        if self.fine_grained:
            img_fine_feat = self.fine_grained_proj_img(image_embeds)[:,1:,:] # [batch, f, embddim]
            txt_fine_feat = self.fine_grained_proj_txt(text_embeds)[:,1:,:] # [batch, w, embddim]
            fine_sim = torch.einsum("bfd,bwd->bfw", [img_fine_feat, txt_fine_feat]) # [batch, f, w]
            txt_max_sim = torch.max(fine_sim, dim=-1)[0] # batch w
            img_max_sim = torch.max(fine_sim, dim=-2)[0] # batch f
            img_fine_feat = self.fine_grained_proj_img2(img_fine_feat).squeeze(-1) # batch f
            txt_fine_feat = self.fine_grained_proj_txt2(txt_fine_feat).squeeze(-1) # batch w
            Si2t = img_max_sim @ txt_fine_feat.t()
            St2i = txt_max_sim @ img_fine_feat.t()
            S_itm = self.norm_itm(Si2t + St2i)
            loss_fine_grained = self.get_fine_grained_loss(S_itm)
            add_loss.update({'loss_fine_grained': loss_fine_grained})
        
        if self.gpt:
            gpt_ids = gpt_input.input_ids
            gpt_atts = gpt_input.attention_mask
            gpt_feat = self.text_proj(self.get_text_embeds(gpt_ids, gpt_atts))[:, 0, :]

            co_gpt_feat1, co_text_feat = self.gpt_caption(gpt_feat, text_feat)
            loss_gtc = self.get_contrastive_loss(co_gpt_feat1, co_text_feat, idx=idx)

            co_gpt_feats2, co_image_feat = self.img_caption(gpt_feat, image_feat)
            loss_g2i = self.get_contrastive_loss(co_gpt_feats2, co_image_feat, idx=idx)
            add_loss.update({'loss_gpt_contrasive': loss_g2i + loss_gtc})


            
                
        return add_loss


