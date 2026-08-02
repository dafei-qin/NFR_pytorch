
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from CNN import LinearWN, TextureEncoder, TextureDecoder
from UNET import UNet
# from pointnet2_utils import PointNetSetAbstraction
import logging
import sys
import time
sys.path.append('./third_party/diffusion-net/src')
import diffusion_net
# from pointnet2_utils import PointNetSetAbstraction
class STN3d(nn.Module):
    def __init__(self, channel, norm=False):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        self.bn4 = nn.Identity()
        self.bn5 = nn.Identity()
        if norm:
            self.bn1 = nn.GroupNorm(4, 64)
            self.bn2 = nn.GroupNorm(8, 128)
            self.bn3 = nn.GroupNorm(8, 1024)
            self.bn4 = nn.GroupNorm(8, 512)
            self.bn5 = nn.GroupNorm(8, 256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, norm=False):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        self.bn4 = nn.Identity()
        self.bn5 = nn.Identity()
        if norm:
            # self.bn1 = nn.GroupNorm(64 // 4, 64)
            # self.bn2 = nn.GroupNorm(128 // 4, 128)
            # self.bn3 = nn.GroupNorm(1024 //4, 1024)
            # self.bn4 = nn.GroupNorm(512 // 4, 512)
            # self.bn5 = nn.GroupNorm(256 // 4, 256)
            self.bn1 = nn.GroupNorm(4, 64)
            self.bn2 = nn.GroupNorm(4, 128)
            self.bn3 = nn.GroupNorm(8, 1024)
            self.bn4 = nn.GroupNorm(8, 512)
            self.bn5 = nn.GroupNorm(8, 256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, reduce=True, out_channel=1024, channel=3, norm=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel, norm)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_channel, 1)
        self.out_channel = out_channel
        if norm:

            self.bn1 = nn.GroupNorm(4, 64)
            self.bn2 = nn.GroupNorm(4, 128)
            self.bn3 = nn.GroupNorm(8, out_channel)

            # self.bn1 = nn.GroupNorm(64 // 4, 64)
            # self.bn2 = nn.GroupNorm(128 // 4, 128)
            # self.bn3 = nn.GroupNorm(out_channel // 4, out_channel)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.reduce = reduce
        if self.feature_transform:
            self.fstn = STNkd(k=64, norm=norm)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.reduce:
            x = torch.max(x, 2, keepdim=True)[0] # Reduce the number of points dimension.
            x = x.view(-1, self.out_channel)
        else:
            x = x
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.out_channel, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda(trans.get_device())
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class transpose(nn.Module):
    def forward(self, x):
        return torch.transpose(x, -2, -1)

class my_diffusion_net(nn.Module):
    def __init__(self, in_shape, out_shape, hid_shape, N_block=4, outputs_at='faces'):
        super(my_diffusion_net, self).__init__()
        self.dfn = diffusion_net.DiffusionNet(C_in=in_shape, C_out=out_shape, C_width=hid_shape, N_block=N_block, outputs_at=outputs_at)
    def forward(self, inputs, dfn_info):
        mass, L, evals, evecs, gradX, gradY, faces = dfn_info
        outputs = self.dfn(inputs, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        return outputs

class my_diffusion_net_template(nn.Module):
    def __init__(self, in_shape, out_shape, hid_shape, pre_computes, N_block=4, outputs_at='faces', with_grad=True):
        super(my_diffusion_net_template, self).__init__()
        self.dfn = diffusion_net.DiffusionNet(C_in=in_shape, C_out=out_shape, C_width=hid_shape, N_block=N_block, outputs_at=outputs_at, with_gradient_features=with_grad)
        self.mass = nn.Parameter(pre_computes[0], requires_grad=False)
        # self.L = pre_computes[1]
        self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
        self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
        self.L_size = pre_computes[1].size()
        self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
        self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
        self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
        self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
        self.grad_X_size =pre_computes[4].size()
        self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
        self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
        self.grad_Y_size = pre_computes[5].size()
        # self.gradX = pre_computes[4]
        # self.gradY = pre_computes[5]
        self.faces = nn.Parameter(pre_computes[6], requires_grad=False)

    def update_precomputes(self, pre_computes):
        self.mass = nn.Parameter(pre_computes[0], requires_grad=False)
        # self.L = pre_computes[1]
        self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
        self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
        self.L_size = pre_computes[1].size()
        self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
        self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
        self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
        self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
        self.grad_X_size =pre_computes[4].size()
        self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
        self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
        self.grad_Y_size = pre_computes[5].size()
        # self.gradX = pre_computes[4]
        # self.gradY = pre_computes[5]
        self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)
        # self.faces = self.faces.unsqueeze(0).long()


    def forward(self, inputs, batch_mass=None, batch_L_val=None, batch_evals=None, batch_evecs=None, batch_gradX=None, batch_gradY=None):
        # print(inputs.device)
        self.L = torch.sparse_coo_tensor(self.L_ind, self.L_val, self.L_size, device=inputs.device)
        batch_size = inputs.shape[0]
        if batch_mass is not None:
            batch_L = [torch.sparse_coo_tensor(self.L_ind, batch_L_val[i], self.L_size, device=inputs.device) for i in range(len(batch_L_val))]
        else:
            
            batch_L = [self.L for b in range(batch_size)]
            batch_mass = self.mass.unsqueeze(0).expand(batch_size, -1)
            batch_evals = self.evals.unsqueeze(0).expand(batch_size, -1)
            batch_evecs = self.evecs.unsqueeze(0).expand(batch_size, -1, -1)
        # self.gradX = torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=inputs.device)
        # self.gradY = torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=inputs.device)
        if batch_gradX is not None:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, gX, self.grad_X_size, device=inputs.device) for gX in batch_gradX ]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, gY, self.grad_Y_size, device=inputs.device) for gY in batch_gradY ]
        else:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=inputs.device) for b in range(batch_size)]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=inputs.device) for b in range(batch_size)]
        outputs = self.dfn(inputs, batch_mass, L=batch_L, evals=batch_evals, evecs=batch_evecs, gradX=gradX, gradY=gradY, faces=self.faces)
        return outputs

    
class Lipschitz_Linear(nn.Linear):
    # Custom Linear layer that use a lipschitz smoothness constraint, see [Liu et al 2022] https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/model.py
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self.c = nn.Parameter(torch.max(torch.sum(torch.abs(self.weight), axis=1)))



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        softplus_c = F.softplus(self.c)
        abs_row_sum = torch.sum(torch.abs(self.weight), axis=1)
        scale = torch.minimum(torch.tensor(1.0), softplus_c / abs_row_sum)

        return F.linear(input, self.weight * scale.unsqueeze(1), self.bias)


class SineLayer(nn.Module):
    # SIREN paper. See https://github.com/vsitzmann/siren
    # Code obtained from their colab notebook: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=hqmlk4AbRleI
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

# class mylinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, is_first=True, omega_0=30):
#         super(mylinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

class mymodel(nn.Module):
    def __init__(self, in_shape, hid_shape=128, use_lipschitz=False):
        super(mymodel, self).__init__()
        if use_lipschitz:
            linear_layer = Lipschitz_Linear # Use the customized lipschitz layer
        else:
            linear_layer = nn.Linear
        self.model = nn.Sequential(
                linear_layer(in_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, 9)
            )

    def __call__(self, input):
        return self.model(input)


class iden_branch(nn.Module):
    def __init__(self, enc_inshape, mlp_inshape, out_shape, pre_computes, hid_shape=128, latent_shape=128, dfn_blocks=4, number_gn=32, iden_encoder='pn', encoder_norm=False):
        super(iden_branch, self).__init__()

        linear_layer = nn.Linear
        self.relu = nn.ReLU()
        self.iden_encoder = iden_encoder
        if self.iden_encoder == 'pn':
            self.encoder = PointNet(enc_inshape, latent_shape, norm=encoder_norm)
        else:
            self.encoder = my_diffusion_net_template(in_shape=enc_inshape, out_shape=latent_shape, hid_shape=hid_shape, pre_computes=pre_computes, N_block=dfn_blocks, outputs_at='global_mean')
        self.linears = [linear_layer(mlp_inshape + latent_shape, hid_shape, bias=False)]
        
        for _ in range(6):
           
            self.linears.append(linear_layer(hid_shape, hid_shape, bias=False))
        
        self.linears = nn.ModuleList(self.linears)
        self.linear_out = linear_layer(hid_shape, out_shape)
        self.gns = [nn.GroupNorm(number_gn, hid_shape) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)

    def update_precomputes(self, dfn_info):
        self.encoder.update_precomputes(dfn_info)

    def forward(self, source_f, target_v):
        N_F = source_f.shape[1]

        # Iden encoder
        if self.iden_encoder == 'pn':
            target_v = target_v.transpose(2, 1)
            latent, pn_trans = self.encoder(target_v)
        else:
            latent = self.encoder(target_v)

        latent = latent.unsqueeze(1)
        latent_toconcat = latent.expand(-1, N_F, -1)

        # MLP
        out = torch.cat([source_f, latent_toconcat], dim=-1)
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)

        out = self.linear_out(out)

        return out, latent



class latent_space(nn.Module):
    def __init__(self, gn_in, in_shape, out_shape, pre_computes=None, hid_shape=128, latent_shape=32, global_pn_shape=None, use_siren=False, encoder='pn',  dfn_blocks=4, iden_encoder='pn', iden_blocks=4, non_linear='relu', residual=False, global_pn=False, sampling=0, pn_norm=False, number_gn=32, code_concat=False, img_encoder=False, img_feat=32, img_norm='bn', img_only_mlp=False, img_warp=False, global_encoder_grad=True, iden_encoder_grad=True):
        super(latent_space, self).__init__()
        linear_layer = nn.Linear
        pn = PointNet

        if non_linear == 'relu':
            self.relu = nn.ReLU()
        elif non_linear == 'leaky_relu':
            self.relu = nn.LeakyReLU()
        else:
            logging.error(f'{non_linear} not implemented! Choices from [relu, leaky_relu]')

        self.code_concat = code_concat
        self.residual = residual
        self.encoder_type = encoder
        self.sampling = sampling
        self.img_enc_type = img_encoder
        self.use_siren = use_siren
        self.img_only_mlp = img_only_mlp
        self.img_warp = img_warp
        self.latent_shape = latent_shape
        # Img encoder
        if self.img_enc_type == 'cnn':
            self.img_encoder = TextureEncoder()
            self.img_fc = linear_layer(128, img_feat)
            if not self.img_only_mlp and not self.img_warp:
                gn_in += img_feat
            if not self.img_warp:
                in_shape += img_feat
        elif self.img_enc_type == 'unet':
            self.img_encoder = UNet(n_channels=4, n_classes=img_feat, norm=img_norm) 
            if not self.img_only_mlp and not self.img_warp:
                gn_in += img_feat
            if not self.img_warp:
                in_shape += img_feat
        else:
            self.img_encoder = None

        # Exp encoder
        if encoder == 'pn':
            self.encoder = pn(gn_in, latent_shape, norm=pn_norm)
        elif encoder == 'dfn':
            self.encoder = my_diffusion_net_template(in_shape=gn_in, out_shape=latent_shape, hid_shape=hid_shape, pre_computes=pre_computes, N_block=dfn_blocks, outputs_at='global_mean', with_grad=global_encoder_grad)
        elif encoder == 'linear':
            self.encoder = linear_layer(in_features=gn_in, out_features=latent_shape)
        # Iden encoder
        self.iden_encoder_type = iden_encoder
        if global_pn:
            if global_pn_shape == None:
                global_pn_shape = latent_shape
            if iden_encoder == 'pn':
                self.global_pn = pn(gn_in, global_pn_shape, norm=pn_norm)
            elif iden_encoder == 'dfn':
                if self.img_warp:
                    self.global_pn = my_diffusion_net_template(in_shape=gn_in, out_shape=global_pn_shape, hid_shape=hid_shape, pre_computes=pre_computes, N_block=iden_blocks, outputs_at='faces', with_grad=iden_encoder_grad)
                else:
                    self.global_pn = my_diffusion_net_template(in_shape=gn_in, out_shape=global_pn_shape, hid_shape=hid_shape, pre_computes=pre_computes, N_block=iden_blocks, outputs_at='global_mean', with_grad=iden_encoder_grad)

        else:
            self.global_pn = None

        # Warper:
        if self.img_warp:
            self.warper = linear_layer(img_feat + global_pn_shape, 3)

        # Useless...
        self.fc_mu = linear_layer(latent_shape, latent_shape)
        self.fc_var = linear_layer(latent_shape, latent_shape)

        # MLP
        if self.use_siren:
            if global_pn and not self.img_warp:
                self.linears = [SineLayer(in_shape+global_pn_shape + latent_shape, hid_shape, bias=True, is_first=True)]
            else:
                self.linears = [SineLayer(in_shape+ latent_shape, hid_shape, bias=True, is_first=True)]
            for _ in range(6):
                self.linears.append(SineLayer(hid_shape, hid_shape, bias=True))
        else:
            if global_pn and not self.img_warp:
                    self.linears = [linear_layer(in_shape+global_pn_shape + latent_shape, hid_shape, bias=False)]
            else:
                    self.linears = [linear_layer(in_shape+ latent_shape, hid_shape, bias=False)]
            for _ in range(6):
                if self.code_concat:
                    self.linears.append(linear_layer(hid_shape + latent_shape, hid_shape, bias=False) )
                else:
                    self.linears.append(linear_layer(hid_shape, hid_shape, bias=False))
            

            self.gns = [nn.GroupNorm(number_gn, hid_shape) for _ in range(len(self.linears))]
            self.gns = nn.ModuleList(self.gns)

        self.linears = nn.ModuleList(self.linears)
        if self.code_concat:
            self.linear_out = linear_layer(hid_shape + latent_shape, out_shape)
        else:
            self.linear_out = linear_layer(hid_shape, out_shape)

    def update_precomputes(self, dfn_info):
        if self.encoder_type == 'dfn':
            self.encoder.update_precomputes(dfn_info)
        if self.iden_encoder_type == 'dfn':
            self.global_pn.update_precomputes(dfn_info)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        See https://github.com/pytorch/examples/blob/main/vae/main.py
        """
        # TODO: add multigpu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to("cuda")  
        return eps * std + mu

    def img_feat(self, img):
        if self.img_enc_type == 'cnn':
            return self.img_fc(self.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
        elif self.img_enc_type == 'unet':
            feat_per_pixel, feat_enc = self.img_encoder(img.permute(0, 3, 1, 2))
            return feat_per_pixel

    def forward(self, x_v, x_f, batch_mass=None, batch_L_val=None, batch_evals=None, batch_evecs=None, batch_gradX=None, batch_gradY=None, source_img=None, target_img=None, return_latent=False, latent=None, return_time=False, x_source_v=None, pix2face_idxs=None, pix2face_vals=None, neutral_pix2face_idxs=None, neutral_pix2face_vals=None, face2vertex_idxs=None, face2vertex_vals=None):

        N_F = x_f.shape[1]
        N_V = x_v.shape[1]
        B_S = x_v.shape[0]
        # x_f = x_f.expand(B_S, -1, -1)
        if latent is not None:
            latent = latent.to(x_f.device)
        if self.img_encoder is not None:
            if not self.img_only_mlp and not self.img_warp:
                img = torch.cat((source_img, target_img), axis=0)
            else:
                img = source_img
            img = img.permute(0, 3, 1, 2) # Use permute to retain the correct order
            if self.img_enc_type == 'cnn':
                img_feat = self.img_encoder(img[:, :3]) # DO NOT CONSIDER DEPTH NOW
                img_feat = self.img_fc(img_feat)
                if not self.img_warp:
                    x_f = torch.cat((x_f, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_F, -1)), dim=-1) # The first index is the source feature
                if not self.img_only_mlp and not self.img_warp:
                    x_v = torch.cat((x_v, img_feat[1:].unsqueeze(1).expand(-1, N_V, -1)), dim=-1) # The followings are the target feature
                    if x_source_v is not None:
                        x_source_v = torch.cat((x_source_v, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_V, -1)), dim=-1)
                img_feat_return = img_feat
            elif self.img_enc_type == 'unet':
                # Construct sparse matrices
                neutral_pix2face = torch.sparse.FloatTensor(neutral_pix2face_idxs[0], neutral_pix2face_vals[0], (N_F, 256**2)).to(img.device)
                face2vertex = torch.sparse.FloatTensor(face2vertex_idxs[0], face2vertex_vals[0], (N_V, N_F)).to(img.device)
                pix2face = torch.cat([torch.sparse.FloatTensor(pix2face_idxs[b], pix2face_vals[b], (N_F, 256**2)).unsqueeze(0) for b in range(B_S)], dim=0).to(img.device)

                feat_per_pixel, feat_enc = self.img_encoder(img) # (B, 128, 256, 256) feature map
                # feat_per_face_batch = []
                feat_per_vertex_batch = []
                feat_per_face_source = torch.sparse.mm(neutral_pix2face, feat_per_pixel[0].reshape(feat_per_pixel.shape[1], -1).transpose(0, 1))
                feat_per_vertex_source = torch.sparse.mm(face2vertex, feat_per_face_source)
                if not self.img_warp:
                    x_f = torch.cat((x_f, feat_per_face_source.unsqueeze(0).expand(-1, -1, -1)), dim=-1)

                if not self.img_only_mlp and not self.img_warp:
                    for b in range(B_S):
                        feat_per_face = torch.sparse.mm(pix2face[b], feat_per_pixel[b+1].reshape(feat_per_pixel.shape[1], -1).transpose(0, 1))
                        feat_per_vertex = torch.sparse.mm(face2vertex, feat_per_face)
                        feat_per_vertex_batch.append(feat_per_vertex.unsqueeze(0))
                    feat_per_vertex_batch = torch.cat(feat_per_vertex_batch, axis=0)
                    x_v = torch.cat([x_v, feat_per_vertex_batch], dim=-1)
                    x_source_v = torch.cat([x_source_v, feat_per_vertex_source.unsqueeze(0)], dim=-1)
                
                img_feat_return = feat_per_pixel
        else:
            img_feat_return = None


        # Expression Encoder:
        t = time.time()
        if self.encoder_type == 'pn':
            if self.sampling != 0:
                p = torch.ones(x_v.shape[1])
                sample_idxs = p.multinomial(num_samples=1000, replacement=False)
                x_v = x_v[:, sample_idxs]
            x_v = x_v.transpose(2, 1)
            pn_feat_return, pn_trans = self.encoder(x_v)
        elif self.encoder_type == 'dfn':
            pn_feat_return = self.encoder(x_v, batch_mass=batch_mass,  batch_L_val=batch_L_val, batch_evals=batch_evals, batch_evecs=batch_evecs, batch_gradX=batch_gradX, batch_gradY=batch_gradY)
            
        elif self.encoder_type == 'linear':
                pn_feat_return = self.encoder(x_v)
                pn_feat_return = pn_feat_return.max(axis=1).values
        if latent is not None:
            pn_feat = latent
        else:
            pn_feat = pn_feat_return
        t_exp = time.time() - t

        # Identity Encoder
        t = time.time()
        if self.global_pn is not None:
            if self.iden_encoder_type == 'pn':
                if self.sampling == 2:
                    p = torch.ones(x_source_v.shape[1])
                    sample_idxs = p.multinomial(num_samples=1000, replacement=False)
                    x_source_v_sampled = x_source_v[:, sample_idxs]
                else:
                    x_source_v_sampled = x_source_v
                source_feat, source_trans = self.global_pn(x_source_v_sampled.transpose(2, 1))
            elif self.iden_encoder_type == 'dfn':
                source_feat = self.global_pn(x_source_v, None, None)
            
            if not self.img_warp:
                source_feat = source_feat.expand(B_S, -1)
                pn_feat = torch.cat([source_feat, pn_feat], dim=-1)
                pn_feat_return = torch.cat([source_feat, pn_feat_return], dim=-1)

        t_iden = time.time() - t

        # Warper
        if self.img_warp:
            warper_in = torch.cat([source_feat, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_F, -1)], dim=-1)
            warp_vec = self.warper(warper_in)
            # x_f[..., :3] = x_f[..., :3] + warp_vec
            x_f = torch.cat((x_f[..., :3] + warp_vec, x_f[..., 3:]), axis=-1)
        
        # MLP
        t = time.time()
        z = pn_feat # This is for direct latent-code
        feat = z.unsqueeze(1)
        feat = feat.expand(-1, N_F, -1) 
        x_f = x_f.expand(B_S, -1, -1)
        x_f = torch.cat([x_f, feat], dim=-1)
        out = x_f


        if self.residual:
            out = torch.transpose(self.relu(self.gns[0](torch.transpose(self.linears[0](out), -1, -2))), -1, -2)
            out_ = torch.transpose(self.relu(self.gns[1](torch.transpose(self.linears[1](out), -1, -2))), -1, -2)
            out_ = torch.transpose(self.linears[2](out), -1, -2)
            out = out.transpose(-1, -2) + out_
            out = torch.transpose(self.relu(self.gns[2](out)), -1, -2)

            out_ = torch.transpose(self.relu(self.gns[3](torch.transpose(self.linears[3](out), -1, -2))), -1, -2)
            out_ = torch.transpose(self.linears[4](out), -1, -2)
            out = out.transpose(-1, -2) + out_
            out = torch.transpose(self.relu(self.gns[4](out)), -1, -2)

            out_ = torch.transpose(self.relu(self.gns[5](torch.transpose(self.linears[5](out), -1, -2))), -1, -2)
            out_ = torch.transpose(self.linears[6](out), -1, -2)
            out = out.transpose(-1, -2) + out_
            out = torch.transpose(self.relu(self.gns[6](out)), -1, -2)

        else:
            # t = time.time()
            if self.use_siren:
                for _ in range(len(self.linears)):
                    out = self.linears[_](out)
            else:
                for _ in range(len(self.linears)):
                    out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)
                    if self.code_concat:
                        out = torch.concat((out, feat[:, :, -53:]), dim=-1)
            # print(f'Calculating: {time.time() - t:.6f}s')
            t_linear = time.time() - t

        out = self.linear_out(out)
        self.cached_feat = feat
        if return_time:
            return out, None, None, pn_feat_return, img_feat_return, np.array([t_exp, t_iden, t_linear])
        else:
            return out, None, None, pn_feat_return, img_feat_return

    def encode(self, x_v, target_img=None, p2f_t_idxs=None, p2f_t_vals=None, f2v_t_idxs=None, f2v_t_vals=None, N_F=None, batch_mass=None, batch_L_val=None, batch_evals=None, batch_evecs=None, batch_gradX=None, batch_gradY=None):
        N_V = x_v.shape[1]
        # CNN
        if not self.img_only_mlp and not self.img_warp:
            if self.img_encoder is not None:
                #img = torch.cat((source_img, target_img), axis=0).transpose(1, 3)
                img = target_img.permute(0, 3, 1, 2)
                if self.img_enc_type == 'cnn':
                    img_feat = self.img_encoder(img[:, :3, ...]) # DO NOT CONSIDER DEPTH NOW
                    img_feat = self.img_fc(img_feat)
                    x_v = torch.cat((x_v, img_feat.unsqueeze(0).expand(-1, N_V, -1)), dim=-1) # The followings are the target feature


                elif self.img_enc_type == 'unet':
                    # Construct sparse matrices
                    face2vertex = torch.sparse.FloatTensor(f2v_t_idxs[0], f2v_t_vals[0], (N_V, N_F)).to(img.device)
                    pix2face = torch.sparse.FloatTensor(p2f_t_idxs[0], p2f_t_vals[0], (N_F, 256**2)).to(img.device)

                    feat_per_pixel, feat_enc = self.img_encoder(img) # (B, 128, 256, 256) feature map
            
                    feat_per_face = torch.sparse.mm(pix2face, feat_per_pixel.reshape(feat_per_pixel.shape[1], -1).transpose(0, 1))
                    feat_per_vertex = torch.sparse.mm(face2vertex, feat_per_face).unsqueeze(0)
                    x_v = torch.cat([x_v, feat_per_vertex], dim=-1)

            

        # Expression Encoder:
        t = time.time()
        if self.encoder_type == 'pn':
            if self.sampling != 0:
                p = torch.ones(x_v.shape[1])
                sample_idxs = p.multinomial(num_samples=1000, replacement=False)
                x_v = x_v[:, sample_idxs]
            x_v = x_v.transpose(2, 1)
            pn_feat_return, pn_trans = self.encoder(x_v)
        elif self.encoder_type == 'dfn' or self.encoder_type == 'linear':
            pn_feat_return = self.encoder(x_v, batch_mass=batch_mass, batch_L_val=batch_L_val, batch_evals=batch_evals, batch_evecs=batch_evecs, batch_gradX=batch_gradX, batch_gradY=batch_gradY)
            if self.encoder_type == 'linear':
                pn_feat_return = pn_feat_return.max(axis=1).values
        self.cached_exp = pn_feat_return
        return pn_feat_return

    def encode_exp(self, x_v):
        return self.encode(x_v)

    def encode_iden(self, x_v):

        if self.global_pn is not None:
            if self.iden_encoder_type == 'pn':
                if self.sampling == 2:
                    p = torch.ones(x_v.shape[1])
                    sample_idxs = p.multinomial(num_samples=1000, replacement=False)
                    x_v_sampled = x_v[:, sample_idxs]
                else:
                    x_v_sampled = x_v
                source_feat, source_trans = self.global_pn(x_v_sampled.transpose(2, 1))
            elif self.iden_encoder_type == 'dfn':
                source_feat = self.global_pn(x_v)
            self.cached_iden = source_feat
            return source_feat
        else:
            logging.error('This model does not contain an identity encoder!')
            raise ValueError
        
    def decode(self, x_f, latent, x_f_orig=None, x_v_orig=None, source_img=None):
        if self.iden_encoder_type == 'dfn':
            x_f, x_source_v = x_f
        N_F = x_f.shape[1]
        B_S = x_f.shape[0]
        # Iden enc
        if latent.shape[1] == self.latent_shape : # Only the exp code is provided
            if self.global_pn and not self.img_warp:
                if self.iden_encoder_type == 'pn':

                    if x_f_orig is not None:
                        latent_iden, _ = self.global_pn(x_f_orig.transpose(2, 1))
                    else:
                        latent_iden, _ = self.global_pn(x_f.transpose(2, 1))
                elif self.iden_encoder_type == 'dfn':
                    if x_v_orig is not None:
                        latent_iden= self.global_pn(x_v_orig)
                    else:
                        latent_iden= self.global_pn(x_source_v)
                    
                latent_all = torch.cat((latent_iden, latent), axis=-1)
            else:
                latent_all = latent
                latent_iden = None
        else: # Fixed both exp and iden code
            latent_iden = latent[:, :-53]
            latent_all = latent
        latent_all = latent_all.unsqueeze(1).expand(B_S, N_F, -1)

        # MLP
        x = torch.cat([x_f, latent_all], dim=-1)
        out = x
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)
            if self.code_concat:
                out = torch.concat((out, latent_all[:, :, -53:]), dim=-1)
        out = self.linear_out(out)
        self.cached_feat = latent_all[:, 0]
        return out, latent_iden
        
    def sample(self, x_f):
        x_f = torch.cat([x_f, self.cached_feat.unsqueeze(1).expand(-1, x_f.shape[1], -1)], dim=-1)
        out = x_f
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)

        out = self.linear_out(out)
        return out
    
class PointNet(nn.Module):
    def __init__(self, channel_in, channel_out, conv_out=1024, reduce=True, norm=False):
        super(PointNet, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, reduce=reduce, channel=channel_in, out_channel=conv_out, norm=norm)
        self.reduce = reduce
        if self.reduce:
            self.fc1 = nn.Linear(conv_out, conv_out // 2)
            self.fc2 = nn.Linear(conv_out // 2, conv_out // 4)
            self.fc3 = nn.Linear(conv_out // 4, channel_out)
        self.dropout = nn.Dropout(p=0.4)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        if norm:
            self.bn1 = nn.GroupNorm(4, conv_out // 2)
            self.bn2 = nn.GroupNorm(4, conv_out // 4)
            # self.bn1 = nn.BatchNorm1d(conv_out //2)
            # self.bn2 = nn.BatchNorm1d(conv_out // 4)
        self.relu = nn.ReLU()

    def update_precomputes(self, _):
        return 

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x, trans_feat



# class pointnet2_ssg(nn.Module):
#     # Borrow heavily from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_ssg.py
#     def __init__(self,in_channel,out_channel, norm=False):
#         super(pointnet2_ssg, self).__init__()

#         self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 256, 512], group_all=True)
#         self.fc1 = nn.Linear(512, 256)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(256, 128)

#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(128, out_channel)
#         if norm:
#             self.bn1 = nn.GroupNorm(256 / 4, 256)
#             self.bn2 = nn.GroupNorm(128 / 4, 128)
#         else:
#             self.bn1 = nn.Identity()
#             self.bn2 = nn.Identity()

#     def forward(self, xyz):
#         B, C, N = xyz.shape

#         if C > 3:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 512)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)


#         return x, l3_points

# class pointnet2_ssg(nn.Module):
#     # Borrow heavily from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_ssg.py
#     def __init__(self,in_channel,out_channel, norm=False):
#         super(pointnet2_ssg, self).__init__()

#         self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 256, 512], group_all=True)
#         self.fc1 = nn.Linear(512, 256)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(256, 128)

#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(128, out_channel)
#         if norm:
#             self.bn1 = nn.GroupNorm(4, 256)
#             self.bn2 = nn.GroupNorm(4, 128)
#             # self.bn1 = nn.BatchNorm1d(256)
#             # self.bn2 = nn.BatchNorm1d(128)
#         else:
#             self.bn1 = nn.Identity()
#             self.bn2 = nn.Identity()

#     def forward(self, xyz):
#         B, C, N = xyz.shape

#         if C > 3:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 512)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)


#         return x, l3_points




class df_net(nn.Module):
    def __init__(self, gn_in, gn_out, mlp_in, mlp_hid, use_gn=True, use_lipschitz=False, global_encoder='pn', gn_hid=128, N_block=4, outputs_at='faces'):
        super(df_net, self).__init__()
        self.ge_type = global_encoder
        self.use_gn = use_gn
        
        if self.use_gn:
            if global_encoder == 'pn':
                self.ge = PointNet(gn_in, gn_out)
            elif global_encoder == 'dfn':
                self.ge = my_diffusion_net(in_shape=gn_in, out_shape=gn_out, hid_shape=gn_hid, N_block=N_block, outputs_at=outputs_at)
            else:
                print(f'Expected pn or dfn as global encoder, but received {global_encoder}')
                raise NotImplementedError
        self.mlp = mymodel(mlp_in, mlp_hid, use_lipschitz=use_lipschitz)


    def forward(self, x_v, x_f):
        # x_v: input features for the global encoder
        # x_f: input features for the mlp
        N_F = x_f.shape[1]
        B_S = x_f.shape[0]
        if self.use_gn:
            if self.ge_type == 'pn':
                x_v = x_v.transpose(2, 1)
                pn_feat, pn_trans = self.ge(x_v)
                feat = pn_feat.unsqueeze(1)
            elif self.ge_type == 'dfn':
                dfn_feat = self.ge(x_v)
                feat = torch.max(dfn_feat, -2, keepdim=True)[0]

            feat = feat.expand(B_S, N_F, -1) 
            x_f = torch.cat([x_f, feat], dim=-1)
        
            # print(dfn_feat.shape)
        out = self.mlp(x_f)
        return out