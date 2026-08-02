
import logging
import os
import sys
from copy import deepcopy
from collections import defaultdict
import warnings
import shutil

import numpy as np

import torch
from torch import nn
from torch.utils.dlpack import from_dlpack
from cupyx.scipy.sparse.linalg import SuperLU
from torch_sparse import coalesce, transpose
from tensorboardX import SummaryWriter
import cupy

sys.path.append('./third_party/diffusion-net/src')
import diffusion_net
from deformation_transfer import Mesh, Transfer, deformation_gradient, deformation_gradient_cho
from mymodel import df_net, latent_space

import time
from FWH_args import train_parser
from data_loader import load_data_ict_all, load_data_ict_random, load_data_ict_live, load_data_mf_v3,load_data_ict_argument
from myutils.myutils import reconstruct_jacobians, positional_encoder, Normalizer, load_state_dict, Normalizer_landmark, Normalizer_img, calc_norm_torch
from datasets import fake_dataset

def calc_loss(idx, v=None, v_gt=None, j=None, j_gt=None, mu=None, log_var=None, latent=None, latent_shape=128, latent_type='facs', reduce=True, v_factor=100, j_factor=1, kld_factor=5e-4, mask=None, latent_const=None, iden_vec=None, latent_factor=2e-2, zero_loss=None, zero_activation=None, global_weights=None, norm=None, norm_gt=None, norm_factor=1, latent_reg_factor=0.01):
    if type(idx) == int:
        idx = [idx]
    if global_weights is None:
        global_weights = torch.ones(len(idx)).unsqueeze(-1).unsqueeze(-1).to(args.device)

    else:
        global_weights = global_weights[idx].unsqueeze(-1).unsqueeze(-1).to(args.device)


    Loss = nn.MSELoss(reduction='none')
    Loss_l1 = nn.L1Loss(reduction='none')
    loss = {'total_loss': torch.tensor(0).to(args.device)}

    if v is not None:
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(-1)
            v = v * mask
            v_gt = v_gt * mask
        vloss = Loss(v, v_gt) * global_weights
        vloss = torch.sqrt(vloss.sum(dim=-1) + 1e-6).mean(dim=1)
        loss['vloss'] = vloss * 100 # for display 
        loss['total_loss'] = loss['total_loss'] + v_factor * vloss

    if j is not None and j_factor != 0:
        jloss = Loss(j, j_gt).view(j.shape[0], j.shape[1],-1)* global_weights
        jloss = jloss.mean(dim=(1, 2))
        loss['jloss'] = jloss
        loss['total_loss'] = loss['total_loss'] + jloss * j_factor

    if norm is not None and norm_factor != 0:
        nloss = Loss(norm, norm_gt) * global_weights
        nloss = nloss.mean(dim=(1, 2))
        loss['nloss'] = nloss
        loss['total_loss'] = loss['total_loss'] + nloss * norm_factor
    if mu is not None:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        loss['kld'] = kld_loss
        loss['total_loss'] += kld_factor * kld_loss

    assert latent_type in ['facs', 'extend']

    if latent_type == 'facs': # Only 53 dimensional facs expression vector
        if type(latent_const) == str or latent_const is None: # Unsupervised
            latent_exp = latent[:, -53:].clone()
            mask_below_zero = latent_exp < 0
            mask_greater_one = latent_exp > 1
            mask_good = ~mask_below_zero * ~mask_greater_one
            latent_loss = latent_exp
            latent_loss[mask_good] = 0
            latent_loss[mask_greater_one] -= 1
            latent_loss[mask_below_zero] *= -1
            latent_loss = latent_loss.mean(dim=-1)
        else:
            # Supervised
            latent_const_ = latent_const[idx].to(latent.device)
            latent_exp = latent[:, -53:]
            latent_loss = Loss(latent_exp, latent_const_).sum(dim=1)

    elif latent_type == 'extend': # Add to 128 dim to allow details
        if latent_const == 'restricted': # Only allow FACS units
            latent_facs = latent[:, -latent_shape: -latent_shape + 53].clone()
            latent_extended = latent[:, -latent_shape+53:]
            mask_below_zero = latent_facs < 0
            mask_greater_one = latent_facs > 1
            mask_good = ~mask_below_zero * ~mask_greater_one
            latent_loss = latent_facs
            latent_loss[mask_good] = 0
            latent_loss[mask_greater_one] -= 1
            latent_loss[mask_below_zero] *= -1
            latent_loss = latent_loss.mean(dim=-1) + Loss(latent_extended, torch.zeros_like(latent_extended).to(latent_extended.device)).mean(dim=-1) # Keep the FACS units in [0, 1] and let the extended terms be 0

        elif latent_const == 'free': # Fully unsupervised
            latent_facs = latent[:, -latent_shape:-latent_shape + 53].clone()
            latent_extended = latent[:, -latent_shape + 53:]
            mask_below_zero = latent_facs < 0
            mask_greater_one = latent_facs > 1
            mask_good = ~mask_below_zero * ~mask_greater_one
            latent_loss = latent_facs
            latent_loss[mask_good] = 0
            latent_loss[mask_greater_one] -= 1
            latent_loss[mask_below_zero] *= -1
            latent_loss = latent_loss.mean(dim=-1) + latent_reg_factor * torch.norm(latent_extended, p=1, dim=-1) # Keep the FACS units in [0, 1] and add l1 regularization to the extended terms

        else: # Supervised
            latent_const_ = latent_const[idx].to(latent.device)
            latent_exp = latent[:, -latent_shape: -latent_shape + 53] # The FACS unit
            latent_extended = latent[:, -latent_shape + 53:] # The extended unit
            latent_loss = Loss(latent_exp, latent_const_).sum(dim=1) + Loss(latent_extended, torch.zeros_like(latent_extended).to(latent_extended.device)).sum(dim=1)

    loss['latent'] = latent_loss
    loss['total_loss'] += latent_factor * latent_loss

    if zero_loss is not None:

        zero_loss = Loss(zero_activation, torch.empty_like(zero_activation)).mean(dim=1)
        loss['zero'] = zero_loss
        loss['total_loss'] += zero_loss

    with torch.no_grad():
        l1_loss = Loss_l1(v, v_gt)  * global_weights 
        l1_loss = l1_loss.mean(dim=(1, 2))
        loss['l1'] =  l1_loss

    if reduce:
        for key in loss.keys():
            loss[key] = loss[key].mean()

    return loss

def eval(model, e, face, val_loaders, dfn_info, transfs, args, normalizer, write=True, shift=None, device='cuda:0', head='val', fraction=20, mask=None, latent_const=None, iden_vec=None, exp_weights=None):
    global writer
    t__ = time.time() # Overall time

    dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    if isinstance(model, nn.DataParallel):
        model.module.update_precomputes(dfn_info)
    else:
        model.update_precomputes(dfn_info)

    if exp_weights is not None and type(exp_weights) == np.ndarray:
        exp_weights = torch.from_numpy(exp_weights)
    if exp_weights is not None:
        exp_weights = exp_weights.to(args.device)
    # if latent_const is not None and type(latent_const) == np.ndarray:
    #     latent_const = torch.from_numpy(latent_const).float()


    # model.eval()
    Loss = nn.MSELoss(reduction='none')
    e = int(e)
    if write:
        if os.path.exists(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}')):
            logging.warning(f"Store directory {os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}')} exists, deleting...")
            shutil.rmtree(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}'))
        
        os.makedirs(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}'), exist_ok=False)

    total_loss = defaultdict(list)
    counter = 0
    t = time.time()
    myfunc = deformation_gradient.apply
    if 'test' in head:
        l2_distance = []
    with torch.no_grad():
        # t_all = np.array([0, 0, 0, 0])
        for out_idx, train_loader in enumerate(val_loaders):
            input = train_loader.dataset.inputs_source_f.to(args.device).unsqueeze(0)
            input_source_v = train_loader.dataset.inputs_source_v.to(args.device).unsqueeze(0)
            face = train_loader.dataset.face
            
            # Randomize dfn operators for DN_e
            if train_loader.dataset.dfn_info_list is not None and args.global_encoder == 'dfn':
                dfn_info = train_loader.dataset.dfn_info_list[torch.randint(len(train_loader.dataset.dfn_info_list), (1,))]
                dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]

            #     if len(args.parallel_ids) > 0:
            #         model.module.encoder.update_precomputes(dfn_info)
            #     else:
            #         model.encoder.update_precomputes(dfn_info)

            # Update dfn operatores for DN
            if train_loader.dataset.dfn_info is not None and args.iden_encoder == 'dfn':
                dfn_info = train_loader.dataset.dfn_info
                # face = train_loader.dataset.face
                dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
                if len(args.parallel_ids) > 0:
                    model.module.global_pn.update_precomputes(dfn_info)
                else:
                    model.global_pn.update_precomputes(dfn_info)
                if not train_loader.dataset.dfn_info_list:
                    if len(args.parallel_ids) > 0:
                        model.module.encoder.update_precomputes(dfn_info)
                    else:
                        model.encoder.update_precomputes(dfn_info)
            
            if train_loader.dataset.dfn_info is not None:
                dfn_info = train_loader.dataset.dfn_info
                face = train_loader.dataset.face
                dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
                if len(args.parallel_ids) > 0:
                    model.module.update_precomputes(dfn_info)
                else:
                    model.update_precomputes(dfn_info)
                    
            if args.img:
                neutral_img = train_loader.dataset.neutral_img.to(args.device).unsqueeze(0)
                if args.pix2face:
                    neutral_pix2face_indices = train_loader.dataset.neutral_pix2face_indices.to(args.device).unsqueeze(0)
                    neutral_pix2face_values = train_loader.dataset.neutral_pix2face_values.to(args.device).unsqueeze(0)
                    face2vertex_indices = train_loader.dataset.face2vertex_indices.to(args.device).unsqueeze(0)
                    face2vertex_values = train_loader.dataset.face2vertex_values.to(args.device).unsqueeze(0)
                else:
                    neutral_pix2face_indices = None
                    neutral_pix2face_values = None
                    face2vertex_indices = None
                    face2vertex_values = None
            else:
                neutral_img, neutral_pix2face_indices, neutral_pix2face_values, face2vertex_indices, face2vertex_values = None, None, None, None, None
            if len(args.parallel_ids) > 0:
                input_source_v = input_source_v.expand(len(args.parallel_ids), -1, -1)
                input = input.expand(len(args.parallel_ids), -1, -1)
                if args.img:
                    neutral_img = neutral_img.expand(len(args.parallel_ids), -1, -1, -1)
                    if args.pix2face:
                        neutral_pix2face_indices = neutral_pix2face_indices.expand(len(args.parallel_ids), -1, -1)
                        neutral_pix2face_values = neutral_pix2face_values.expand(len(args.parallel_ids), -1)
                        
                        face2vertex_indices = face2vertex_indices.expand(len(args.parallel_ids), -1, -1)
                        face2vertex_values = face2vertex_values.expand(len(args.parallel_ids), -1)
                    else:
                        neutral_pix2face_indices, neutral_pix2face_values, face2vertex_indices, face2vertex_values = None, None, None, None
            loader_fake = torch.utils.data.DataLoader(fake_dataset(len(train_loader.dataset)), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

            for idx in loader_fake:
                input_v = train_loader.dataset.inputs_target_v[idx].to(args.device)
                g = train_loader.dataset.gradients[idx].to(args.device)
                out_gt = train_loader.dataset.verts[idx].to(args.device)
                norm_gt = train_loader.dataset.norms[idx].to(args.device)
                if args.img:
                    imgs = train_loader.dataset.img[idx].to(args.device)

                    if args.pix2face:
                        pix2face_idxs = train_loader.dataset.pix2face_indices[idx]
                        pix2face_vals = train_loader.dataset.pix2face_values[idx]
                    else:
                        pix2face_idxs = None
                        pix2face_vals = None

                else:
                    imgs = None
                    pix2face_idxs = None
                    pix2face_vals = None
                    
                if train_loader.dataset.gradX is not None:
                    gradX_batch = train_loader.dataset.gradX[idx].to(args.device)
                    gradX_batch.requires_grad = False
                    gradY_batch = train_loader.dataset.gradY[idx].to(args.device)
                    gradY_batch.requires_grad = False
                else:
                    gradX_batch = None
                    gradY_batch = None
                counter +=1
                if counter % fraction:
                    continue
                if type(input) == list:
                    N_FACE = input[0].shape[1]
                else:
                    N_FACE = input.shape[1]
                N_VERTEX = out_gt.shape[1]

                # if len(input.shape) == 4:
                #     input = input[0].to(device)
                #     g = g.squeeze(0).to(device)
                #     out_gt = out_gt[0].to(device)


                # if len(input_v.shape) == 4:
                #         input_v = input_v[0].to(device)
                # else:
                #         input_v = input_v.to(device)

                if not args.ab_vertex:
                    g = normalizer.inv_normalize(g)

                    out_gt = out_gt - out_gt.mean(axis=1, keepdim=True)
                    if 'test' in head:
                        write_idx = shift['test'][out_idx]
                    elif 'val' in head:
                        write_idx = shift['val'][out_idx]
                    elif 'train' in head:
                        write_idx = shift['train'][out_idx]
                    else:
                        logging.error(f'Evaluation must provide a head to contain [train, val, test] to identify the transfs_shift, but f{head} is provided.')
                        raise ValueError

                    transf = transfs[write_idx]

                    solver = transf.solver
                    idxs, vals = coalesce(from_dlpack(transf.idxs.toDlpack()).long(), from_dlpack(transf.vals.toDlpack()), m=N_FACE *3, n=N_VERTEX)
                    idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)

                    g = reconstruct_jacobians(g)

                    out_gt_g = myfunc(g, solver, idxs, vals, (N_VERTEX, N_FACE * 3))
                    out_gt_g = out_gt_g - out_gt_g.mean(axis=1, keepdim=True)
                    out_gt = out_gt.to(device)

                    try:
                        assert Loss(out_gt, out_gt_g).mean() < 1e-8
                    except AssertionError:
                        logging.error('Assertion Error! Reconstruction of ground truth jacobians have big errors, please check transf')
                        if not os.path.exists(os.path.join(args.save_dir, args.save_head+f'{e:04d}_error')):
                            os.makedirs(os.path.join(args.save_dir, args.save_head+f'{e:04d}_error'))
                        for exp_idx in range(out_gt.shape[0]):
                            mesh_out = Mesh(out_gt[exp_idx].detach().cpu().numpy(), face)
                            mesh_out.write(os.path.join(args.save_dir, args.save_head+f'{e:04d}_error', f"{write_idx:03d}_{exp_idx:02d}_gt.obj"))
                            mesh_out = Mesh(out_gt_g[exp_idx].detach().cpu().numpy(), face)
                            mesh_out.write(os.path.join(args.save_dir, args.save_head+f'{e:04d}_error', f"{write_idx:03d}_{exp_idx:02d}.obj"))
                        exit(-1)
                
                # Do some argumentation & add noises

                if args.shift_noise:
                    with torch.no_grad():
                        input[:, :, :3] = input[:, :, :3] + torch.randn(3).to(input.device) * 0.1
                if args.teacher_forcing:
                    if latent_const is None or type(latent_const) == str:
                        logging.error('Teacher Forcing requires gt expression vectors')
                        raise ValueError
                    latent_t_f = latent_const[idx].to(input_v.device)
                    if args.teacher_forcing_noise:
                        latent_t_f += (torch.randn(latent_t_f.shape, device=latent_t_f.device) * 1e-3)
                    if args.latent_type == 'extend':
                        latent_t_f = torch.cat([latent_t_f, torch.zeros((latent_t_f.shape[0], args.feat_dim - 53)).to(latent_t_f.device)], dim=-1)

                    # g_pred, mu, log_var, latent, t = model(input_v, input, latent=latent_t_f)
                    g_pred, mu, log_var, latent, latent_img = model(input_v, input, x_source_v=input_source_v, latent=latent_t_f, source_img=neutral_img, target_img=imgs, pix2face_idxs=pix2face_idxs, pix2face_vals=pix2face_vals, neutral_pix2face_idxs=neutral_pix2face_indices, neutral_pix2face_vals=neutral_pix2face_values, face2vertex_idxs=face2vertex_indices, face2vertex_vals=face2vertex_values, batch_gradX=gradX_batch, batch_gradY=gradY_batch)
                    # t_all = t_all +  t
                else:
                    g_pred, mu, log_var, latent, latent_img = model(input_v, input, x_source_v=input_source_v, source_img=neutral_img, target_img=imgs, pix2face_idxs=pix2face_idxs, pix2face_vals=pix2face_vals, neutral_pix2face_idxs=neutral_pix2face_indices, neutral_pix2face_vals=neutral_pix2face_values, face2vertex_idxs=face2vertex_indices, face2vertex_vals=face2vertex_values, batch_gradX=gradX_batch, batch_gradY=gradY_batch)

                if torch.isnan(g_pred).any():
                    print()

                if not args.ab_vertex:
                    g_pred = normalizer.inv_normalize(g_pred)

                    g_pred = reconstruct_jacobians(g_pred)
                    if torch.isnan(g_pred).any():
                        print()
                    out_pred = myfunc(g_pred, solver, idxs, vals, (N_VERTEX, N_FACE * 3))
                else:
                    out_pred = normalizer.inv_normalize(g_pred)

                out_pred = out_pred -  out_pred.mean(axis=1, keepdim=True)


                if iden_vec is not None:
                    iden_vec_ = iden_vec[write_idx]
                else:
                    iden_vec_ = None
                if args.norm_factor != 0:
                    norm = calc_norm_torch(out_pred, face)
                else:
                    norm = None
                loss = calc_loss(idx, out_pred, out_gt, g_pred, g, mu, log_var, latent, reduce=False, v_factor=args.v_factor, j_factor=args.j_factor, kld_factor=args.kld_factor, mask=mask, latent_const=latent_const, latent_factor=args.latent_factor, iden_vec=iden_vec_, global_weights=exp_weights, norm=norm, norm_gt=norm_gt, norm_factor=args.norm_factor, latent_type=args.latent_type, latent_shape=args.feat_dim, latent_reg_factor=args.latent_reg_factor)

                if 'test' in head:
                    l2_distance.append(torch.sqrt(torch.sum((out_pred - out_gt)**2, dim=2)))
                for loss_key in loss.keys():
                    total_loss[loss_key].append(loss[loss_key].mean().item())

                if write:
                    if latent_img is not None:
                        latent_img_source=  latent_img[0].detach().cpu().numpy()
                        np.save(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_neutral_img.npy"), (((latent_img_source - latent_img_source.min()) / (latent_img_source.max()- latent_img_source.min())) * 255).astype(np.uint8))

                    for _, exp_idx in enumerate(idx):
                        mesh_out = Mesh(out_pred[_].detach().cpu().numpy(), face)
                        mesh_out.write(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_" + ''.join([f'{key}_{value[_]:.4f}_' for key, value in loss.items()]) + '.obj'))
                        mesh_gt = Mesh(out_gt[_].detach().cpu().numpy(), face)
                        mesh_gt.write(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_gt.obj"))

                        if args.save_exps:
                            latent_ = latent[_].detach().cpu().numpy()
                            np.save(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_vec.npy"), latent_)
                            if latent_img is not None:
                                if latent_img.shape[0] > 1:
                                    latent_img_=  latent_img[_ + 1].detach().cpu().numpy()
                                else:
                                    latent_img_ = latent_img[0].cpu().numpy()
                                if len(latent_img_.shape) == 3: # Full img
                                    np.save(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_img.npy"), (((latent_img_ - latent_img_.min()) / (latent_img_.max()- latent_img_.min())) * 255).astype(np.uint8))
                                else:
                                    np.save(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_img.npy"), latent_img_)
                            if latent_const is not None and type(latent_const) != str:
                                latent_gt = latent_const[exp_idx].detach().cpu().numpy()
                                np.save(os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}', f"{out_idx:04d}_{exp_idx:04d}_expvec_gt.npy"), latent_gt)


    logging.info(f" [{args.save_head}] Epoch {e} done. {head}: " + ''.join([f'{key}: {np.mean(value):.6f}-{np.std(value):.6f}-{np.median(value):.6f} ' for key, value in total_loss.items()]) + f't: {time.time()-t__:.2f}s')
    if 'test' in head:
        l2_distance = torch.tensor([_.mean() for _ in l2_distance])
        logging.info(f" [{args.save_head}] Epoch {e} done. {head} l2 distance: {torch.mean(l2_distance.view(-1, )):.6f}-{torch.std(l2_distance.view(-1, )):.6f}-{torch.median(l2_distance.view(-1, )):.6f}")
    for key, value in total_loss.items():
        writer.add_scalar(f'{head}-loss/{key}', np.mean(value), e)

    model.train()

def train(model, train_loaders, val_loaders, dfn_info, transfs, face, args, normalizer, transf_shift, latent_const=None, latent_const_val=None, iden_vec=None, exp_weights = None, epoch=None, head='train', test_loaders=None, fix_mlp=False):
    global writer, optim, scheduler, myfunc, Loss


    dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    if isinstance(model, nn.DataParallel):
        model.module.update_precomputes(dfn_info)
        for name, param in model.module.named_parameters():
            if any(_ in name for _ in ['mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X_ind', 'grad_X_val', 'grad_Y_ind', 'grad_Y_val', 'faces']):
                param.requires_grad = False
            else:
                param.requires_grad = True
        if fix_mlp:
            for param in model.module.linears.parameters():
                param.requires_grad = False
            for param in model.module.linear_out.parameters():
                param.requires_grad= False
            for param in model.module.gns.parameters():
                param.requires_grad = False
    else:
        model.update_precomputes(dfn_info)
        for name, param in model.named_parameters():
            if any(_ in name for _ in ['mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X_ind', 'grad_X_val', 'grad_Y_ind', 'grad_Y_val', 'faces']):
                param.requires_grad = False
            else:
                param.requires_grad = True
        if fix_mlp:
            for param in model.linears.parameters():
                param.requires_grad = False
            for param in model.linear_out.parameters():
                param.requires_grad= False
            for param in model.gns.parameters():
                param.requires_grad = False
    
    # print(head)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # return 
    if exp_weights is not None and type(exp_weights) == np.ndarray:
        exp_weights = torch.from_numpy(exp_weights).float()
    if exp_weights is not None:
        exp_weights = exp_weights.to(args.device)

    if epoch == None:
        resume_epoch = args.resume_id
        epoch_to_run = args.EPOCH
    else:
        epoch_to_run = 0
        resume_epoch = epoch
    
    for e in range(resume_epoch, resume_epoch + epoch_to_run + 1):
        # print(e)
        total_loss = defaultdict(list)
        tt = time.time()
        if args.debug:
            t_ = time.time()
            t_data_prepare = 0
            t_data_load = 0
            t_model_detail = np.array([0, 0, 0])
            t_model = 0
            t_df = 0
            t_bk = 0
            t_grad = 0
            t_render = 0
        if args.write:
            save_dir = os.path.join(args.save_dir, args.save_head+f'{e:04d}_{head}_train')
                                    
            if os.path.exists(save_dir):
                logging.warning(f"Store directory {save_dir} exists, deleting...")
                shutil.rmtree(save_dir)
        
            os.makedirs(save_dir)
        for out_idx, train_loader in enumerate(train_loaders):
            face = train_loader.dataset.face
            input = train_loader.dataset.inputs_source_f.to(args.device).unsqueeze(0)
            input_source_v = train_loader.dataset.inputs_source_v.to(args.device).unsqueeze(0)
            # Randomize dfn operators for DN_e, not using
            if train_loader.dataset.dfn_info_list is not None and args.global_encoder == 'dfn':
                dfn_info = train_loader.dataset.dfn_info_list[torch.randint(len(train_loader.dataset.dfn_info_list), (1,))]
                dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
                # face = train_loader.dataset.face

                if len(args.parallel_ids) > 0:
                    model.module.encoder.update_precomputes(dfn_info)
                else:
                    model.encoder.update_precomputes(dfn_info)

            # Update dfn operatores for DN
            if train_loader.dataset.dfn_info is not None and args.iden_encoder == 'dfn':
                dfn_info = train_loader.dataset.dfn_info
                
                dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
                if len(args.parallel_ids) > 0:
                    model.module.global_pn.update_precomputes(dfn_info)
                else:
                    model.global_pn.update_precomputes(dfn_info)
                if not train_loader.dataset.dfn_info_list:
                    if len(args.parallel_ids) > 0:
                        model.module.encoder.update_precomputes(dfn_info)
                    else:
                        model.encoder.update_precomputes(dfn_info)

            if args.img:
                neutral_img = train_loader.dataset.neutral_img.to(args.device).unsqueeze(0)
                if args.pix2face:
                    neutral_pix2face_indices = train_loader.dataset.neutral_pix2face_indices.to(args.device).unsqueeze(0)
                    neutral_pix2face_values = train_loader.dataset.neutral_pix2face_values.to(args.device).unsqueeze(0)
                    face2vertex_indices = train_loader.dataset.face2vertex_indices.to(args.device).unsqueeze(0)
                    face2vertex_values = train_loader.dataset.face2vertex_values.to(args.device).unsqueeze(0)
                else:
                    # neutral_img = None
                    neutral_pix2face_indices = None
                    neutral_pix2face_values = None
                    face2vertex_indices = None
                    face2vertex_values = None
            else:
                neutral_img, neutral_pix2face_indices, neutral_pix2face_values, face2vertex_indices, face2vertex_values = None, None, None, None, None
            if len(args.parallel_ids) > 0:
                input_source_v = input_source_v.expand(len(args.parallel_ids), -1, -1)
                input = input.expand(len(args.parallel_ids), -1, -1)
                if args.img:
                    neutral_img = neutral_img.expand(len(args.parallel_ids), -1, -1, -1)
                    if args.pix2face:
                        neutral_pix2face_indices = neutral_pix2face_indices.expand(len(args.parallel_ids), -1, -1)
                        neutral_pix2face_values = neutral_pix2face_values.expand(len(args.parallel_ids), -1)
                        
                        face2vertex_indices = face2vertex_indices.expand(len(args.parallel_ids), -1, -1)
                        face2vertex_values = face2vertex_values.expand(len(args.parallel_ids), -1)

            loader_fake = torch.utils.data.DataLoader(fake_dataset(len(train_loader.dataset)), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

            for idx in loader_fake:
                input_v = train_loader.dataset.inputs_target_v[idx].to(args.device)
                g = train_loader.dataset.gradients[idx].to(args.device)
                out_gt = train_loader.dataset.verts[idx].to(args.device)
                norm_gt = train_loader.dataset.norms[idx].to(args.device)
                if args.img:
                    imgs = train_loader.dataset.img[idx].to(args.device)

                    if args.pix2face:
                        pix2face_idxs = train_loader.dataset.pix2face_indices[idx]
                        pix2face_vals = train_loader.dataset.pix2face_values[idx]

                    else:
                        pix2face_idxs = None
                        pix2face_vals = None
                else:
                    imgs, pix2faces = None, None
                    pix2face_idxs = None
                    pix2face_vals = None
                if train_loader.dataset.gradX is not None:
                    gradX_batch = train_loader.dataset.gradX[idx].to(args.device)
                    # gradX_batch.requires_grad = False
                    gradY_batch = train_loader.dataset.gradY[idx].to(args.device)
                    # gradY_batch.requires_grad = False
                else:
                    gradX_batch = None
                    gradY_batch = None
                    
                N_VERTEX = out_gt.shape[1]
                N_FACE = input.shape[1]

                if args.debug:
                    t_data_prepare += (time.time() - t_)
                    # logging.debug(f'Data preparing: {time.time() - t:.2f}s')
                    t_ = time.time()


                g = g.squeeze().to(args.device)
                g = normalizer.inv_normalize(g)
                if len(out_gt.shape) == 4:
                    out_gt = out_gt.squeeze()
                out_gt = out_gt.to(args.device)
                out_gt = out_gt - out_gt.mean(axis=1, keepdim=True)
                if 'test' in head:
                    transf = transfs[transf_shift['test'][out_idx]]
                elif 'val' in head:
                    transf = transfs[transf_shift['val'][out_idx]]
                else:
                    transf = transfs[transf_shift['train'][out_idx]]

                if args.debug:
                    t_data_load += (time.time() - t_)
                    t_ = time.time()

                solver = transf.solver
                idxs, vals = coalesce(from_dlpack(transf.idxs.toDlpack()).long(), from_dlpack(transf.vals.toDlpack()), m=N_FACE *3, n=N_VERTEX)
                idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)
                g = reconstruct_jacobians(g)
                
                if e == 0:
                    Loss = torch.nn.MSELoss()
                    with torch.no_grad():

                        out_gt_g = myfunc(g, solver, idxs, vals, (N_VERTEX, N_FACE * 3))
                        out_gt_g = out_gt_g - out_gt_g.mean(axis=1, keepdim=True)
                    try:
                        assert Loss(out_gt, out_gt_g) < 1e-7
                    except AssertionError:
                        logging.info(f'The l2 difference between ground truth and transfered: {Loss(out_gt, out_gt_g)} is bigger than theroshold 1e-7!')   
                        if len(idx) == 1:
                            write_idx = idx
                        else:
                            write_idx = out_idx

                        for exp_idx in range(out_gt.shape[0]):
                            with torch.no_grad():
                                mesh = Mesh(out_gt_g[exp_idx].detach().cpu().numpy(), face)
                                mesh_gt = Mesh(out_gt[exp_idx].detach().cpu().numpy(), face)
                                logging.info(exp_idx, Loss(out_gt_g[exp_idx], out_gt[exp_idx]).detach().item())
                                mesh.write(f'./{write_idx}_{exp_idx}_transf_{Loss(out_gt[exp_idx], out_gt_g[exp_idx]).item()}.obj')
                                mesh_gt.write(f'./{write_idx}_{exp_idx}_gt.obj')
                                dg_temp = g[exp_idx].detach().cpu().numpy()
                                np.save(f'./{write_idx}_{exp_idx}_g.npy', dg_temp)
                        logging.info(f'idx: {write_idx} All saves to ./[idx]_[exp_idx]_[gt/transf].obj')
                        logging.info(f'Abort.')
                        exit(1)

                for param in model.parameters():
                    param.grad = None

                if args.teacher_forcing:
                    if latent_const is None or type(latent_const) == str:
                        logging.error('Teacher Forcing requires gt expression vectors')
                        raise ValueError
                    latent_t_f = latent_const[idx].to(args.device)
                    if args.teacher_forcing_noise:
                        latent_t_f += (torch.randn(latent_t_f.shape, device=latent_t_f.device) * 1e-3)
                    if args.latent_type == 'extend':
                        latent_t_f = torch.cat([latent_t_f, torch.zeros((latent_t_f.shape[0], args.feat_dim - 53)).to(latent_t_f.device)], dim=-1)

                    g_pred, mu, log_var, latent, latent_img= model(input_v, input, latent=latent_t_f, x_source_v=input_source_v, source_img=neutral_img, target_img=imgs, pix2face_idxs=pix2face_idxs, pix2face_vals=pix2face_vals, neutral_pix2face_idxs=neutral_pix2face_indices, neutral_pix2face_vals=neutral_pix2face_values, face2vertex_idxs=face2vertex_indices, face2vertex_vals=face2vertex_values,return_time=False, batch_gradX=gradX_batch, batch_gradY=gradY_batch)

                    # t_all = t_all +  t_model_np
                else:

                    g_pred, mu, log_var, latent, latent_img = model(input_v, input, x_source_v=input_source_v, source_img=neutral_img, target_img=imgs,pix2face_idxs=pix2face_idxs, pix2face_vals=pix2face_vals, neutral_pix2face_idxs=neutral_pix2face_indices, neutral_pix2face_vals=neutral_pix2face_values, face2vertex_idxs=face2vertex_indices, face2vertex_vals=face2vertex_values,return_time=False, batch_gradX=gradX_batch, batch_gradY=gradY_batch)


                if args.debug:

                    t_model += (time.time() - t_)
                    t_ = time.time()

                    if torch.isnan(g_pred).any():
                        print()
                    if torch.isnan(g_pred).any():
                        logging.error('Found Nan in g_pred directly after model output!')
                        ckpt = {
                            'epoch': e,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'lr_sched': scheduler.state_dict(),
                            'args': args
                        }
                        torch.save(ckpt, os.path.join(args.save_dir, args.save_head + f'{e}.pth'))
                        logging.info(f"Save checkpoints to {os.path.join(args.save_dir, args.save_head + f'{e}.pth')}")
                        raise ValueError


                g_pred = normalizer.inv_normalize(g_pred)
                g_pred = reconstruct_jacobians(g_pred)

                if args.debug:
                    if torch.isnan(g_pred).any():
                        logging.error('Found Nan in g_pred directly after reconstruction!')
                        ckpt = {
                            'epoch': e,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'lr_sched': scheduler.state_dict(),
                            'args': args
                        }
                        torch.save(ckpt, os.path.join(args.save_dir, args.save_head + f'{e}.pth'))
                        logging.info(f"Save checkpoints to {os.path.join(args.save_dir, args.save_head + f'{e}.pth')}")
                        raise ValueError
                out_pred = myfunc(g_pred, solver, idxs, vals, (N_VERTEX, N_FACE * 3))
                out_pred = out_pred -out_pred.mean(axis=1, keepdim=True)

                if args.debug:
                    # logging.debug(f'DF running: {time.time() - t:.4f}')
                    t_df += (time.time() - t_)
                    t_ = time.time()

                if iden_vec is not None:
                    iden_vec_ = iden_vec[out_idx] # The current identity
                else:
                    iden_vec_ = None
                if args.norm_factor != 0:
                    norm = calc_norm_torch(out_pred, face)
                else:
                    norm = None

                loss = calc_loss(idx, out_pred, out_gt, g_pred, g, None, log_var, latent, v_factor=args.v_factor, j_factor=args.j_factor, kld_factor=args.kld_factor, mask=None, latent_const=latent_const, latent_factor=args.latent_factor, iden_vec=iden_vec_, zero_loss=None, zero_activation=None, global_weights=exp_weights, norm=norm, norm_gt=norm_gt, norm_factor=args.norm_factor, latent_type=args.latent_type, latent_shape=args.feat_dim, latent_reg_factor=args.latent_reg_factor)

                loss['total_loss'].backward()

                if args.debug:
                    t_bk += (time.time()-  t_)
                    t_ = time.time()
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.grad.isnan().any():
                            logging.error(f'Nan found in gradients of parameter: {name}')
                            ckpt = {
                                'epoch': e,
                                'model': model.state_dict(),
                                'optim': optim.state_dict(),
                                'lr_sched': scheduler.state_dict(),
                                'args': args
                            }
                            torch.save(ckpt, os.path.join(args.save_dir, args.save_head + f'{e}.pth'))
                            logging.info(f"Save checkpoints to {os.path.join(args.save_dir, args.save_head + f'{e}.pth')}")
                            raise ValueError
                optim.step()
                for key, value in loss.items():
                    total_loss[key].append(value.mean().item())
                # logging.debug(f'Gradient updating {time.time() - t:.4f}')
                if args.debug:
                    t_grad += (time.time() - t_)
                

                max_grads = [x.grad.abs().max().item() for x in list(model.parameters()) if x.grad is not None]
                avg_grads = [x.grad.abs().mean().item() for x in list(model.parameters()) if x.grad is not None]
                if torch.isnan(torch.mean(torch.tensor(avg_grads))):
                    print()

                t_ = time.time()
                if args.write:
                    for _, exp_idx in enumerate(idx):
                        mesh_out = Mesh(out_pred[_].detach().cpu().numpy(), face)
                        mesh_out.write(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:05d}_" + '.obj'))
                        mesh_gt = Mesh(out_gt[_].detach().cpu().numpy(), face)
                        mesh_gt.write(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:05d}_gt.obj"))
                        if args.save_exps:
                            latent_ = latent[_].detach().cpu().numpy()
                            np.save(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:04d}_vec.npy"), latent_)
                            # if latent_img is not None:
                            #     if latent_img.shape[0] > 1:
                            #         latent_img_=  latent_img[_ + 1].detach().cpu().numpy()
                            #     else:
                            #         latent_img_ = latent_img[0].cpu().numpy()
                            #     if len(latent_img_.shape) == 3: # Full img
                            #         np.save(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:04d}_img.npy"), (((latent_img_ - latent_img_.min()) / (latent_img_.max()- latent_img_.min())) * 255).astype(np.uint8))
                            #     else:
                            #         np.save(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:04d}_img.npy"), latent_img_)
                            if latent_const is not None and type(latent_const) != str:
                                latent_gt = latent_const[exp_idx].detach().cpu().numpy()
                                np.save(os.path.join(save_dir, f"{out_idx:04d}_{exp_idx:04d}_expvec_gt.npy"), latent_gt)
        # Finished an epoch
        scheduler.step()
        if 'test' not in head:
            logging.info(f" [{args.save_head}] Epoch {e} done. {head} " + ''.join([f'{key}: {np.mean(value):.4f} ' for key, value in total_loss.items()])  + f"max_grad: {np.max(max_grads):.4f}, avg_grad: {np.mean(avg_grads):.4f},  lr: {scheduler.get_last_lr()}, t: {time.time() - tt:.4f}s")
        else:
             logging.info(f" [{args.save_head}] Epoch {e} done. {head} " + ''.join([f'{key}: {np.mean(value):.4f} ' for key, value in total_loss.items()])  + f"t: {time.time() - tt:.4f}s")
        if args.debug:
            print(f't_data_prep: {t_data_prepare:.3f}s | t_data_load: {t_data_load:.3f}s | t_model: {t_model:.3f}s | t_df: {t_df:.3f}s | t_bk: {t_bk:.3f}s | t_step: {t_grad:.3f}s |t_render: {t_render:.3f}s| total: {t_data_load + t_data_prepare + t_model + t_df + t_grad + t_render:.3f}s')
            print(f''.join([f'{name}:{ttt:.2f}s ' for name, ttt in zip(['exp', 'iden','linear'], t_model_detail)]))

        for key, value in total_loss.items():
            writer.add_scalar(f"{head}-loss/{key}", np.mean(value), e)
        if 'test' not in head:
            writer.add_scalar(f'{head}/max_grad', np.max(max_grads), e)
            writer.add_scalar(f'{head}/avg_grad', np.mean(avg_grads), e)
            writer.add_scalar('lr/lr', scheduler.get_last_lr(), e)
            writer.add_scalar(f'{head}/t', time.time() - tt, e)

        if not e % args.eval_freq:
            ckpt = {
                'epoch': e,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_sched': scheduler.state_dict(),
                'args': args
            }
            torch.save(ckpt, os.path.join(args.save_dir, args.save_head + f'{e}.pth'))
            eval(model, e, face, val_loaders, dfn_info, transfs, args,  normalizer, write=False, shift=transf_shift, device=args.device, fraction=1, latent_const=latent_const_val, iden_vec=iden_vec,  exp_weights=exp_weights, head=head+'_val')
            eval(model, e, face, train_loaders, dfn_info, transfs, args,  normalizer, write=False, shift=transf_shift, device=args.device, head=head + '_train', fraction=10, exp_weights=exp_weights, latent_const=latent_const, iden_vec=iden_vec)
            if test_loaders is not None:
                eval(model, e, face, test_loaders, dfn_info, transfs, args,  normalizer, write=False, shift=transf_shift, device=args.device, fraction=1, latent_const=latent_const_val, iden_vec=iden_vec,  exp_weights=exp_weights, head=head+'_test')
        if not e % 10:
            ckpt = {
                'epoch': e,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_sched': scheduler.state_dict(),
                'args': args
            }
            torch.save(ckpt, os.path.join(args.save_dir, args.save_head + f'{e}.pth'))



if __name__ == '__main__':
    use_clearml = False
    os.environ['OMP_NUM_THREADS'] = '8'
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.multiprocessing.set_start_method('spawn')
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    parser = train_parser()
    args = parser.parse_args()
    if use_clearml:
        from clearml import Task
        task = Task.init(project_name='NFR_biwi', task_name=args.save_head)
    if args.iden_encoder == 'dfn':
        args.use_source_v = True
    else:
        args.use_source_v = False
    args.save_dir = os.path.join(args.save_dir, args.save_head)
    log_file = os.path.join(args.save_dir, 'log.txt')
    args.save_head = args.save_head + '_'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, format="[%(levelname)s] %(funcName)s: %(message)s", level=logging.DEBUG, force=True)
    writer = SummaryWriter(log_dir=log_file.replace('.txt', '.board'))

    logging.info(f'logging to {log_file}')
    logging.info(f'Dataset: {args.dataset}')
    if args.ab_vertex:
        logging.info(f'WARNING, ablation on jacobians activated.')
        if args.feature_type != 'cents&norms':
            logging.error(f'When ablate jacobians, inputs_v can only be cents&norms! Abort')
            raise ValueError

    t = time.time()
    latent_const = None
    exp_weights = None
    pre_computes = None

    normalizer = Normalizer(args.std_file, args.device)
    if args.wks:
        wks_normalizer = Normalizer(args.wks_std_file, args.device, zero_mean=False)
    else:
        wks_normalizer = None
    if args.landmark:
        landmark_normalizer = Normalizer_landmark(args.landmark_file, args.device)
    else:
        landmark_normalizer = None
    if args.img:
        img_normalizer = Normalizer_img(args.img_file, args.device)
    else:
        img_normalizer = None
    # renderer = renderer()
    with cupy.cuda.Device(int(args.device[-1])):
        if args.dataset == 'MF':
            if args.data_head == 'MF_all_v5' or args.data_head == 'MF_all_v4':
                n_mf = 11
                shuffle = False
                per_loader_dfn_info = False
                transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf), 'test': range(n_mf)}

            elif args.data_head == 'MF_shift_scale':
                n_mf = 100
                shuffle = True
                per_loader_dfn_info = True
                transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf, n_mf+10), 'test': range(n_mf+10, n_mf+20)}
            elif args.data_head == 'MF_holes':
                n_mf = 100
                shuffle = True
                per_loader_dfn_info = True
                transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf, n_mf+10), 'test': range(n_mf+10, n_mf+20)}
            elif args.data_head == 'COMA_no_inner':
                n_mf = 12
                shuffle = False
                per_loader_dfn_info = False
            face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf, dfn_info_mf = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_mf, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, shuffle=shuffle, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf = {'train': range(len(train_loaders_mf)), 'val': range(len(val_loaders_mf)), 'test': range(len(test_loaders_mf))}
            dfn_info = dfn_info_mf
        elif args.dataset == 'COMA':
            n_coma = 1
            face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf, dfn_info_mf = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_coma, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, shuffle=False, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            dfn_info = dfn_info_mf
            transfs_shift_mf = {'train': range(n_coma), 'val': range(n_coma), 'test': range(n_coma)}

        elif args.dataset == 'ICT_single':
            exp_weights, latent_const, iden_vec, face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf,  dfn_info_mf = load_data_ict_all(args.data_dir, args.data_head, args.batch_size, 10, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf = {'train': range(len(train_loaders_mf)), 'val': range(len(train_loaders_mf), len(train_loaders_mf) + len(val_loaders_mf)), 'test': range(len(train_loaders_mf) + len(val_loaders_mf), len(train_loaders_mf) + len(val_loaders_mf) + len(test_loaders_mf))}
            latent_const = torch.from_numpy(latent_const).float()
            dfn_info = dfn_info_mf
            # latent_const = train_loaders[0].dataset.exp_vec        
        elif args.dataset == 'ICT_random':
            latent_const, iden_vec, face, transfs, train_loaders, val_loaders, test_loaders,  dfn_info,  inputs_shape, inputs_v_shape = load_data_ict_random(args.data_dir, args.data_head, args.batch_size, args.eval_range[1], normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, use_hks=args.hks, zero_mean=args.zero_mean, only_test=args.test, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)

            # latent_const = train_loaders[0].dataset.exp_vec
        elif args.dataset == 'ICT_live':
            iden_vec_live, latent_const_live, latent_const_val_live, latent_const_test_live, face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf,  dfn_info_mf = load_data_ict_live(args.data_dir, args.data_head, args.batch_size, 10, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf = {'train': range(len(train_loaders_mf)), 'val': range(len(train_loaders_mf), len(train_loaders_mf) + len(val_loaders_mf)), 'test': range(len(train_loaders_mf) + len(val_loaders_mf), len(train_loaders_mf) + len(val_loaders_mf) + len(test_loaders_mf))}
            latent_const = torch.from_numpy(latent_const_live).float()
            dfn_info = dfn_info_mf
            latent_const_live = torch.from_numpy(latent_const_live).float()
            latent_const_val_live = torch.from_numpy(latent_const_val_live).float()
            latent_const_test_live = torch.from_numpy(latent_const_test_live).float()
            
        elif args.dataset == 'ICT_all':
            if args.debug:
                n_single = 1
                n_random = 1
                n_live = 4
                n_holes = 1
                n_shift_scale = 1
            else:
                n_single = 10
                n_random = 6
                n_live = 10
                n_holes = 100
                n_shift_scale = 100
            args.data_head = 'ICT_single_coarse' # Single expression activations
            exp_weights, latent_const, iden_vec, face, transfs, train_loaders, val_loaders, test_loaders,  dfn_info = load_data_ict_all(args.data_dir, args.data_head, args.batch_size, n_single, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift = {'train': range(len(train_loaders)), 'val': range(len(train_loaders), len(train_loaders) + len(val_loaders)), 'test': range(len(train_loaders) + len(val_loaders), len(train_loaders) + len(val_loaders) + len(test_loaders))}
            latent_const = torch.from_numpy(latent_const).float()
            args.data_head = 'ICT_random_50' # Random expression activations
            latent_const_, iden_vec_, face, transfs_, train_loaders_, val_loaders_, test_loaders_,  dfn_info_ = load_data_ict_random(args.data_dir, args.data_head, args.batch_size, n_random, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_ = {'train': range(len(train_loaders_)), 'val': [len(train_loaders_)], 'test': [len(train_loaders_) + 1]}
            latent_const_ = torch.from_numpy(latent_const_).float()
            args.data_head = 'ICT_live_100' # Live captured expression activations
            iden_vec_live, latent_const_live, latent_const_val_live, latent_const_test_live, face, transfs_live, train_loaders_live, val_loaders_live, test_loaders_live,  dfn_info_live = load_data_ict_live(args.data_dir, args.data_head, args.batch_size, n_live, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_live = {'train': range(len(train_loaders_live)), 'val': range(len(train_loaders_live), len(train_loaders_live) + len(val_loaders_live)), 'test': range(len(val_loaders_live), len(val_loaders_live) + len(test_loaders_live))}
            latent_const_live = torch.from_numpy(latent_const_live).float()
            latent_const_val_live = torch.from_numpy(latent_const_val_live).float()
            latent_const_test_live = torch.from_numpy(latent_const_test_live).float()
            
            args.data_head = 'ICT_shift_scale' # Shift and scale augmented expression activations
            latent_const_ss, face_ss, transfs_ss, train_loaders_ss, val_loaders_ss, test_loaders_ss,  dfn_info_ss = load_data_ict_argument(args.data_dir, args.data_head, args.batch_size, n_shift_scale, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_ss = {'train': range(len(train_loaders_ss)), 'val': range(len(train_loaders_ss), len(train_loaders_ss) + 10), 'test': range(len(train_loaders_ss)+10, len(train_loaders_ss)+20)}
            latent_const_ss = torch.from_numpy(latent_const_ss).float()
            args.data_head = 'ICT_holes' # Holes augmented expression activations
            latent_const_holes, face_holes, transfs_holes, train_loaders_holes, val_loaders_holes, test_loaders_holes,  dfn_info_holes = load_data_ict_argument(args.data_dir, args.data_head, args.batch_size, n_holes, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_holes = {'train': range(len(train_loaders_holes)), 'val': range(len(train_loaders_holes), len(train_loaders_holes) + 10), 'test': range(len(train_loaders_holes)+10, len(train_loaders_holes)+20)}
            # dfn_info = dfn_info_holes
            latent_const_holes = torch.from_numpy(latent_const_holes).float()
        elif args.dataset == 'ALL' or args.dataset == 'MF_all':
            if args.debug:
                n_live = 4
                n_mf = 1
                n_coma = 1
                n_ict_ss = 1
                n_mf_ss = 1
                n_mf_holes=  1
            else:
                n_live = 20
                n_mf = 11
                n_coma = 12
                n_ict_ss = 100
                n_mf_ss = 100
                n_mf_holes = 100
                if args.test:
                    n_live = 1
                    n_mf = 11
                    n_coma = 1
                    n_ict_ss = 1
                    n_mf_ss = 1
                    n_mf_holes = 1
            args.data_head = 'ICT_live_100'
            iden_vec_live, latent_const_live, latent_const_val_live, latent_const_test_live, face, transfs_live, train_loaders_live, val_loaders_live, test_loaders_live,  dfn_info_live = load_data_ict_live(args.data_dir, args.data_head, args.batch_size, n_live, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_live = {'train': range(len(train_loaders_live)), 'val': range(len(train_loaders_live), len(train_loaders_live) + len(val_loaders_live)), 'test': range(len(val_loaders_live), len(val_loaders_live) + len(test_loaders_live))}
            latent_const_live = torch.from_numpy(latent_const_live).float()
            latent_const_val_live = torch.from_numpy(latent_const_val_live).float()
            latent_const_test_live = torch.from_numpy(latent_const_test_live).float()
            dfn_info = dfn_info_live
            
            
            args.data_head = 'MF_all_v5' # Multiface activations
            # args.test = False
            face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf, dfn_info_mf = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_mf, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf), 'test': range(n_mf)}
            args.data_head = 'ICT_shift_scale'
            latent_const_ss, face_ss, transfs_ss, train_loaders_ss, val_loaders_ss, test_loaders_ss,  dfn_info_ss = load_data_ict_argument(args.data_dir, args.data_head, args.batch_size, n_ict_ss, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type, global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_ss = {'train': range(len(train_loaders_ss)), 'val': range(len(train_loaders_ss), len(train_loaders_ss) + 10), 'test': range(len(train_loaders_ss)+10, len(train_loaders_ss)+20)}
            latent_const_ss = torch.from_numpy(latent_const_ss).float()
            args.data_head = 'MF_shift_scale'
            face_mf_ss, transfs_mf_ss, train_loaders_mf_ss, val_loaders_mf_ss, test_loaders_mf_ss, dfn_info_mf_ss = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_mf_ss, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info,  grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf_ss = {'train': range(len(train_loaders_mf_ss)), 'val': range(len(train_loaders_mf_ss), len(train_loaders_mf_ss) + 10), 'test': range(len(train_loaders_mf_ss) + 10, len(train_loaders_mf_ss) + 20)}
            args.data_head = 'MF_holes'
            face_mf_holes, transfs_mf_holes, train_loaders_mf_holes, val_loaders_mf_holes, test_loaders_mf_holes, dfn_info_mf_holes = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_mf_holes, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info,  grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf_holes = {'train': range(len(train_loaders_mf_holes)), 'val': range(len(train_loaders_mf_holes), len(train_loaders_mf_holes) + 10), 'test': range(len(train_loaders_mf_holes) + 10, len(train_loaders_mf_holes) + 20)}


        elif args.dataset == 'biwi':
            if args.debug:
                n_mf = 1
                n_live = 4
            else:
                n_live = 20
                n_mf = 14
            # args.test = False
            args.data_head = 'ICT_live_100'
            iden_vec_live, latent_const_live, latent_const_val_live, latent_const_test_live, face, transfs_live, train_loaders_live, val_loaders_live, test_loaders_live,  dfn_info_live = load_data_ict_live(args.data_dir, args.data_head, args.batch_size, n_live, normalizer,  use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_live = {'train': range(len(train_loaders_live)), 'val': range(len(train_loaders_live), len(train_loaders_live) + len(val_loaders_live)), 'test': range(len(val_loaders_live), len(val_loaders_live) + len(test_loaders_live))}
            latent_const_live = torch.from_numpy(latent_const_live).float()
            latent_const_val_live = torch.from_numpy(latent_const_val_live).float()
            latent_const_test_live = torch.from_numpy(latent_const_test_live).float()
            dfn_info = dfn_info_live
            args.data_head = 'biwi_ssp_8000' # biwi with 8k vertices
            face_mf, transfs_mf, train_loaders_mf, val_loaders_mf, test_loaders_mf, dfn_info_mf = load_data_mf_v3(args.data_dir, args.data_head, args.batch_size, n_mf, normalizer,   use_wks=args.wks, wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=args.test, use_f=True, use_source_v=args.use_source_v, use_landmarks=args.landmark, landmark_normalizer=landmark_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, random_dfn_info=args.random_dfn_info, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
            transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf), 'test': range(n_mf)}

        else:
            logging.error(f'Not implemented datset: {args.dataset}')
            raise NotImplementedError

        logging.info(f'Data loaded, takes up {time.time() - t:.2f}s')


        # DFN_e
        global_encoder_in_shape = 6 if args.feature_type == 'cents&norms' else 12
        if args.wks:
            global_encoder_in_shape += 10
        if args.landmark:
            global_encoder_in_shape += 68 * 2
        # MLP & DFN_i
        in_shape = 6 # cents&norms
        if args.wks:
            in_shape += 10
        if args.landmark:
            in_shape += 68 * 2
        out_shapes = {'matrix': 9, '6dof': 12, 'quat': 10, 'expmap': 9, 'verts':3}
        if args.ab_vertex:
            args.repr = 'verts'

        if dfn_info is not None:
            dfn_info = [_.to(args.device) if type(_) is not torch.Size else _  for _ in dfn_info ]


        logging.info(f'Exp enc: {args.global_encoder}\nExp enc inshape: {global_encoder_in_shape}\nIden enc: {args.iden_encoder}\nIden enc inshape: {in_shape}\n')
        model = latent_space(global_encoder_in_shape, in_shape=in_shape,  out_shape=9, pre_computes=dfn_info, latent_shape=args.feat_dim, encoder=args.global_encoder, iden_encoder=args.iden_encoder, iden_blocks=args.iden_blocks, hid_shape=args.mlp_hid_channel,  non_linear=args.non_linear, residual=False, global_pn=args.global_pn, sampling=args.sampling, pn_norm=args.pn_norm, number_gn=args.num_gn, dfn_blocks=args.dfn_blocks, global_pn_shape=args.global_pn_shape, code_concat=args.code_concat, img_encoder=args.img_enc, img_norm=args.img_norm, use_siren=args.use_siren, img_feat=args.img_feat, img_only_mlp=args.img_only_mlp, img_warp=args.img_warp, global_encoder_grad= not args.no_global_encoder_grad, iden_encoder_grad= not args.no_iden_encoder_grad)

        if len(args.parallel_ids) > 0:
            logging.info(f"Let's use {len(args.parallel_ids)} GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model, device_ids=args.parallel_ids)
                    
        if not args.test:
            model.to(args.device)
            model.float()
            optim = torch.optim.Adam([{'params':model.parameters(), 'initial_lr': args.initial_lr}], lr=args.initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.75, last_epoch=args.resume_id)
            myfunc = deformation_gradient.apply

            if args.resume_id > 0:
                ckpt = torch.load(os.path.join(args.save_dir, args.save_head + f'{args.resume_id}.pth'), map_location=args.device)
                model_state_dict = ckpt['model']
                model = load_state_dict(model, model_state_dict)

                if args.load_lr:
                    optim.load_state_dict(ckpt['optim'])
                    scheduler.load_state_dict(ckpt['lr_sched'])

            if args.pretrained_global_pn:
                ckpt_global_pn = torch.load(args.pretrained_global_pn_dir)
                model.global_pn.load_state_dict(ckpt_global_pn['model'])

            iden_vec = None # just dont use this now 
            if args.dataset == 'ALL' or args.dataset == 'ICT_all' or args.dataset == 'biwi':
                for e in range(args.resume_id, args.EPOCH+args.resume_id):
                    if args.dataset == 'ICT_all':
                        train(model, train_loaders, val_loaders, dfn_info, transfs, face, args, normalizer, transfs_shift, latent_const=latent_const, latent_const_val=latent_const, iden_vec=iden_vec, epoch=e, head='ict_single', exp_weights=exp_weights)

                        train(model, train_loaders_, val_loaders_,  dfn_info_, transfs_, face, args, normalizer, transfs_shift_, latent_const=latent_const_, latent_const_val=latent_const_, iden_vec=None, epoch=e, head='ict_random', exp_weights=None)

                        train(model, train_loaders_live, val_loaders_live, dfn_info_live, transfs_live, face, args, normalizer, transfs_shift_live, latent_const=latent_const_live, latent_const_val=latent_const_val_live, iden_vec=None, epoch=e, head='ict_live', exp_weights=None)

                        train(model, train_loaders_ss, val_loaders_ss, dfn_info_ss, transfs_ss, face_ss, args, normalizer, transfs_shift_ss, latent_const=latent_const_ss, latent_const_val=latent_const_ss, iden_vec=None, epoch=e, head='ict_ss', exp_weights=None)
                        train(model, train_loaders_holes, val_loaders_holes, dfn_info_holes, transfs_holes, face_holes, args, normalizer, transfs_shift_holes, latent_const=latent_const_holes, latent_const_val=latent_const_holes, iden_vec=None, epoch=e, head='ict_holes', exp_weights=None)
                    elif args.dataset == 'ALL':
                        
                        train(model, train_loaders_live, val_loaders_live, dfn_info_live, transfs_live, face, args, normalizer, transfs_shift_live, latent_const=latent_const_live, latent_const_val=latent_const_val_live, iden_vec=None, epoch=e, head='ict_live', exp_weights=None)

                        train(model, train_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=e, head='mf', exp_weights=None, fix_mlp=args.fix_mlp)

                        train(model, train_loaders_ss, val_loaders_ss, dfn_info_ss, transfs_ss, face_ss, args, normalizer, transfs_shift_ss, latent_const=latent_const_ss, latent_const_val=latent_const_ss, iden_vec=None, epoch=e, head='ict_ss', exp_weights=None)

                        train(model, train_loaders_mf_ss, val_loaders_mf_ss, dfn_info_mf_ss, transfs_mf_ss, face_mf_ss, args, normalizer, transfs_shift_mf_ss, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, iden_vec=None, epoch=e, head='mf_ss', exp_weights=None, fix_mlp=args.fix_mlp)

                        train(model, train_loaders_mf_holes, val_loaders_mf_holes, dfn_info_mf_holes, transfs_mf_holes, face_mf_holes, args, normalizer, transfs_shift_mf_holes, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, iden_vec=None, epoch=e, head='mf_holes', exp_weights=None, fix_mlp=args.fix_mlp)

                    elif args.dataset == 'MF_all':
                        train(model, train_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=e, head='mf', exp_weights=None, fix_mlp=args.fix_mlp)

                        train(model, train_loaders_mf_ss, val_loaders_mf_ss, dfn_info_mf_ss, transfs_mf_ss, face_mf_ss, args, normalizer, transfs_shift_mf_ss, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, iden_vec=None, epoch=e, head='mf_ss', exp_weights=None, fix_mlp=args.fix_mlp)

                        train(model, train_loaders_mf_holes, val_loaders_mf_holes, dfn_info_mf_holes, transfs_mf_holes, face_mf_holes, args, normalizer, transfs_shift_mf_holes, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, iden_vec=None, epoch=e, head='mf_holes', exp_weights=None, fix_mlp=args.fix_mlp)

                    elif args.dataset == 'biwi':
                        train(model, train_loaders_live, val_loaders_live, dfn_info_live, transfs_live, face, args, normalizer, transfs_shift_live, latent_const=latent_const_live, latent_const_val=latent_const_val_live, iden_vec=None, epoch=e, head='ict_live', exp_weights=None)
                        
                        train(model, train_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=e, head='biwi', exp_weights=None, fix_mlp=args.fix_mlp)

            else:
                for e in range(args.resume_id, args.EPOCH+args.resume_id):
                    train(model, train_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=e, head='mf', exp_weights=None, test_loaders=test_loaders_mf, fix_mlp = args.fix_mlp)
                    
        else:
            logging.info('Do testing')
            ckpt = torch.load(os.path.join(args.save_dir, args.save_head + f'{args.resume_id}.pth'), map_location=args.device)
            state_dict = ckpt['model']
            model.float()
            myfunc = deformation_gradient.apply
            model = load_state_dict(model, state_dict)
            model.to(args.device)
            optim = torch.optim.Adam([{'params':model.parameters(), 'initial_lr': args.initial_lr}], lr=args.initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.75, last_epoch=args.resume_id)
            
            if args.dataset == 'ICT_all':
                eval(model, args.resume_id, face, test_loaders, dfn_info, transfs, args, normalizer, write=args.write, shift=transfs_shift, device=args.device, head='test_ict_single', fraction=1, mask=None, latent_const=latent_const, iden_vec=None)
                eval(model, args.resume_id, face, test_loaders_, dfn_info_, transfs_, args, normalizer, write=args.write, shift=transfs_shift_, device=args.device, head='test_ict_random', fraction=1, mask=None, latent_const=latent_const_, iden_vec=None)
                eval(model, args.resume_id, face, test_loaders_live, dfn_info_live, transfs_live, args, normalizer, write=args.write, shift=transfs_shift_live, device=args.device, head='test_ict_live', fraction=1, mask=None, latent_const=latent_const_test_live, iden_vec=None)
                eval(model, args.resume_id, face, test_loaders_ss, dfn_info_ss, transfs_ss, args, normalizer, write=args.write, shift=transfs_shift_ss, device=args.device, head='test_ict_ss', fraction=1, mask=None, latent_const=latent_const_ss, iden_vec=None)
                eval(model, args.resume_id, face, test_loaders_holes, dfn_info_holes, transfs_holes, args, normalizer, write=args.write, shift=transfs_shift_holes, device=args.device, head='test_ict_holes', fraction=1, mask=None, latent_const=latent_const_holes, iden_vec=None)

            elif args.dataset == 'ALL':

                # eval(model, args.resume_id, face, test_loaders_live, dfn_info_live, transfs_live, args, normalizer, write=args.write, shift=transfs_shift_live, device=args.device, head='test_ict_live', fraction=1, mask=None, latent_const=latent_const_test_live, iden_vec=None)

                # eval(model, args.resume_id, face_mf, test_loaders_mf, dfn_info_mf, transfs_mf, args, normalizer, write=args.write, shift=transfs_shift_mf,  device=args.device, head='test_mf', fraction=1, mask=None, latent_const=args.latent_loss_type, iden_vec=None)
                # eval(model, args.resume_id, face_mf, val_loaders_mf, dfn_info_mf, transfs_mf, args, normalizer, write=args.write, shift=transfs_shift_mf,  device=args.device, head='mf_val', fraction=1, mask=None, latent_const=args.latent_loss_type, iden_vec=None)
                # eval(model, args.resume_id, face_mf, train_loaders_mf, dfn_info_mf, transfs_mf, args, normalizer, write=args.write, shift=transfs_shift_mf,  device=args.device, head='mf_train', fraction=1, mask=None, latent_const=args.latent_loss_type, iden_vec=None)
                # train(model, train_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head='mf', exp_weights=None, fix_mlp=args.fix_mlp)
                # train(model, val_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head='mf', exp_weights=None, fix_mlp=args.fix_mlp)
                train(model, test_loaders_mf, test_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head='mf_test', exp_weights=None, fix_mlp=args.fix_mlp)
                train(model, test_loaders_live, test_loaders_live, dfn_info_live, transfs_live, face, args, normalizer, transfs_shift_live, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head='ict_live_test', exp_weights=None, fix_mlp=args.fix_mlp)
                
                # eval(model, e, face, val_loaders, dfn_info, transfs, args,  normalizer, write=False, shift=transf_shift, device=args.device, fraction=1, latent_const=latent_const_val, iden_vec=iden_vec,  exp_weights=exp_weights, head=head+'_val')
                # eval(model, args.resume_id, face_mf, val_loaders_mf, dfn_info_mf, transfs_mf, args, normalizer, write=args.write, shift=transfs_shift_mf,  device=args.device, head='val_mf', fraction=1, mask=None, latent_const=None, iden_vec=None)

                # eval(model, args.resume_id, face_coma, test_loaders_coma, dfn_info_coma, transfs_coma, args, normalizer, write=args.write, shift=transfs_shift_coma, device=args.device, head='test_coma', fraction=1, mask=None, latent_const=None, iden_vec=None)

            elif args.dataset == 'ICT_live':
                train(model, test_loaders_mf, test_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=latent_const_test_live, latent_const_val=latent_const_test_live, epoch=int(args.resume_id) + 1, head=args.data_head+'_test', exp_weights=None, fix_mlp=args.fix_mlp)   
            elif args.dataset == 'biwi':
                train(model, test_loaders_mf, test_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=latent_const_test_live, latent_const_val=latent_const_test_live, epoch=int(args.resume_id) + 1, head=args.data_head+'_test', exp_weights=None, fix_mlp=args.fix_mlp)   
            else:      
                train(model, test_loaders_mf, test_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head=args.data_head+'_test', exp_weights=None, fix_mlp=args.fix_mlp)   
                # train(model, test_loaders_mf, val_loaders_mf, dfn_info_mf, transfs_mf, face_mf, args, normalizer, transfs_shift_mf, latent_const=args.latent_loss_type, latent_const_val=args.latent_loss_type, epoch=int(args.resume_id) + 1, head=args.data_head+'_test', exp_weights=None, fix_mlp=args.fix_mlp)   

   