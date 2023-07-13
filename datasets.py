import torch
import numpy as np
from myutils import convert_per_face_to_per_vertex, face_to_vertex_torch
from trimesh import geometry
import igl
import os
import pickle
from glob import glob
from tqdm import tqdm
# from torch.nn.utils.rnn import pad_sequence
def get_centroid(fvs):
    return fvs.mean(axis=-2)

def get_normal(fvs):
    if len(fvs.shape) == 3:
        span = fvs[:, 1:] - fvs[:, :1]
        norm = torch.cross(span[:, 0], span[:, 1])
        return norm / torch.norm(norm, dim=-1, keepdim=True)
    else:
        span = fvs[:, :, 1:] - fvs[:, :, :1]
        norm = torch.cross(span[:, :, 0], span[:, :, 1])
        return norm / torch.norm(norm, dim=-1, keepdim=True)


class fake_dataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.len = length
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return idx

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, face, neutral_verts, verts, gradients, normalizer, neutral_wks=None, wks=None, wks_normalizer=None, neutral_landmarks=None, landmarks=None, landmark_normalizer=None, img=None, neutral_img=None, img_normalizer=None, device='cuda', feature_type='cents&norms',ab_vertex=False,  use_f=True, use_source_v=False, only_iden=False, exp_per_iden=1, pix2face_indices=None, pix2face_values=None,neutral_pix2face=None, dfn_info=None, dfn_info_list=None, gradX=None, gradY=None):

        self.face = torch.from_numpy(face)
        self.verts = verts
        self.neutral_verts = torch.from_numpy(neutral_verts).float()
        self.neutral_fvs = self.neutral_verts[self.face]
        self.neutral_norms = get_normal(self.neutral_fvs)
        self.neutral_cents = get_centroid(self.neutral_verts[self.face].unsqueeze(0)).squeeze()
        self.dfn_info=dfn_info
        self.dfn_info_list=dfn_info_list
        self.len = len(self.verts)
        self.use_f = use_f
        self.use_source_v = use_source_v
        self.device = device
        self.use_pix2face = False
        self.gradX = gradX
        self.gradY = gradY
        
        self.use_wks, self.wks, self.neutral_wks, self.per_face_wks = None, None, None, None
        self.use_landmarks, self.landmarks, self.neutral_landmarks = None, None, None
        self.use_img, self.img, self.neutral_img, self.pix2face, self.neutral_pix2face, self.face2vertex = None, None, None, None, None, None
        # Not used
        if wks is not None:
            self.use_wks = True
            self.wks = wks_normalizer.normalize(torch.from_numpy(wks).float()) / 2
            self.neutral_wks = (wks_normalizer.normalize(torch.from_numpy(neutral_wks).float()) / 2).squeeze()
            self.per_face_wks = self.neutral_wks[self.face].mean(axis=-2)

        if landmarks is not None:
            self.use_landmarks = True
            self.landmarks = landmark_normalizer.normalize(torch.from_numpy(landmarks.reshape(landmarks.shape[0], -1)).float()) / 2
            self.neutral_landmarks = landmark_normalizer.normalize(torch.from_numpy(neutral_landmarks.reshape(-1)).float()) / 2
        
        # Using img
        if img is not None:
            self.use_img = True
            # self.img = img_normalizer.normalize(torch.from_numpy(img)).float()
            # self.neutral_img = img_normalizer.normalize(torch.from_numpy(neutral_img)).float()
            self.img = torch.from_numpy(img).float()
            self.img[self.img == -1] = self.img.amax(dim=(0, 1, 2))[-1]  * 2 # Mapping the zbuf with -1 to the farest
            self.img = self.img / img_normalizer.gradients_std.to(self.img.device)

            self.neutral_img = torch.from_numpy(neutral_img).float()
            self.neutral_img[self.neutral_img == -1] = self.neutral_img.amax(dim=(0, 1))[-1] * 2
            self.neutral_img = self.neutral_img / img_normalizer.gradients_std.to(self.img.device)
            if neutral_pix2face is not None:
                self.use_pix2face = True
                self.pix2face_indices = pix2face_indices.long()
                self.pix2face_values = pix2face_values.float()
                self.neutral_pix2face_indices = neutral_pix2face._indices().long()
                self.neutral_pix2face_values = neutral_pix2face._values().float()
                face2vertex = face_to_vertex_torch(self.face).float()
                self.face2vertex_indices = face2vertex._indices().long()
                self.face2vertex_values = face2vertex._values().float()

        # Inputs_target

        fvs = torch.cat([v[self.face][np.newaxis] for v in self.verts], axis=0) # Face vertices
        self.norms = get_normal(fvs)
        self.norms = self.norms.float()
        self.norms_v = np.concatenate([geometry.mean_vertex_normals(self.verts.shape[1], self.face, n)[np.newaxis] for n in self.norms], axis=0)
        self.norms_v = torch.from_numpy(self.norms_v).float()
        assert not torch.isnan(self.norms_v).any()
        self.verts = self.verts.float()

        # Only for identity branch, not implemented yet: select the neutral and expand
        if only_iden:
            raise NotImplementedError
            self.verts = self.verts[::exp_per_iden]
            self.verts = self.verts.unsqueeze(1).expand(-1, exp_per_iden, -1, -1)
            self.verts = self.verts.reshape(-1, self.norms_v.shape[1], 3)
            

        if feature_type == 'cents&norms':
            self.inputs_target_v = torch.cat([self.verts, self.norms_v], dim=-1)

        # Previous version using the jacobians as input.
        elif feature_type =='jacobians':
            self.inputs_target_v = convert_per_face_to_per_vertex(gradients, self.face, self.verts.shape[1])
            self.inputs_target_v = torch.cat((verts, self.inputs_target_v), dim=-1)

        else:
            raise NotImplementedError

        

        # Inputs_source (Neutral)
        if use_f: # Needs loss on jacobians
            self.gradients = gradients
            self.gradients = normalizer.normalize(self.gradients)
            if only_iden:
                raise NotImplementedError
                self.gradients = self.gradients[::exp_per_iden]
                self.gradients = self.gradients.unsqueeze(1).expand(-1, exp_per_iden, -1, -1)
                self.gradients = self.gradients.reshape(-1, self.norms.shape[1], 9)
            
            # Per face
            self.inputs_source_f = torch.cat([self.neutral_cents, self.neutral_norms], dim=-1) 
            if self.use_wks:
                self.inputs_source_f = torch.cat([self.inputs_source_f, self.per_face_wks], dim=-1)
            if self.use_landmarks:
                self.inputs_source_f = torch.cat([self.inputs_source_f, self.neutral_landmarks.unsqueeze(0).expand(self.inputs_source_f.shape[0], -1)], dim=-1)


        # Per vertex
        self.inputs_source_v = torch.cat((self.neutral_verts, torch.from_numpy(igl.per_vertex_normals(self.neutral_verts.numpy(), self.face.numpy()))), dim=-1) 
        if self.use_wks:
            self.inputs_source_v = torch.cat([self.inputs_source_v, self.neutral_wks], dim=-1)
        if self.use_landmarks:
            self.inputs_source_v = torch.cat([self.inputs_source_v, self.neutral_landmarks.unsqueeze(0).expand(self.inputs_source_v.shape[0], -1)], dim=-1)


    def __len__(self):
        return self.verts.shape[0]


    def __getitem__(self, idx):
    
        inputs_target_v = self.inputs_target_v[idx]
        verts = self.verts[idx]

        if self.use_wks:
            wks = self.wks[idx]
            inputs_target_v = torch.cat([inputs_target_v, wks], dim=-1)
        if self.use_landmarks:
            landmarks = self.landmarks[idx]
            inputs_target_v = torch.cat([inputs_target_v, landmarks.unsqueeze(0).expand(inputs_target_v.shape[0], -1)], dim=-1)


        if self.use_f:
            gradients = self.gradients[idx]
            if self.use_img:
                if self.use_pix2face:
                    return idx,  inputs_target_v.to(self.device).float(), gradients.to(self.device).float(), verts.to(self.device).float(), self.img[idx].to(self.device), self.pix2face[idx].to(self.device)
                else:
                    return idx, inputs_target_v.to(self.device).float(), gradients.to(self.device).float(), verts.to(self.device).float(), self.img[idx].to(self.device)
                
            return idx, inputs_target_v.to(self.device).float(), gradients.to(self.device).float(), verts.to(self.device).float()
        else:
            if self.use_img:
                if self.use_pix2face:
                    return idx, inputs_target_v.to(self.device).float(), verts.to(self.device).float(), self.img[idx], self.pix2face[idx]
                else:
                    return idx, inputs_target_v.to(self.device).float(), verts.to(self.device).float(), self.img[idx]

            return idx, inputs_target_v.to(self.device).float(), verts.to(self.device).float()


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, datapath, device='cuda', use_pix2face=False, img_normalizer=None):
        files = glob(os.path.join(datapath, '*.obj'))
        self.verts = []
        self.face = []
        self.img= []
        self.norms = []
        self.norms_v = []
        self.dfn_info_list = []
        self.inputs_target_v = []
        self.pix2face_indices = []
        self.pix2face_values = []
        self.gradX = None
        self.gradY = None
        for f in tqdm(files):

            with open(f.replace('.obj', '.pkl'), 'rb') as ff:
                verts, face, dfn_info=  pickle.load(ff).values()
            verts = torch.from_numpy(verts).float()
            verts -= verts.mean(axis=(0, 1))
            face = torch.from_numpy(face).long()
            img = np.load(f.replace('.obj', '_img.npy'))
            indices, values, zbuf = torch.load(f.replace('.obj', '_img.pt')).values()

            img = np.concatenate([img, zbuf.unsqueeze(-1).numpy()], axis=-1)
            img = torch.from_numpy(img).float()
            img[img == -1] = torch.amax(img, dim=(0, 1))[-1]  * 2
            # self.img[..., -1] = self.img.amax(dim=(0, 1, 2))[-1] - self.img[..., -1]
            img = img / img_normalizer.gradients_std.to(img.device)
            self.img.append(img.cpu().float())

            self.verts.append(verts)
            self.face.append(face)
            dfn_info = [_.to('cpu') if type(_) is not torch.Size else _  for _ in dfn_info]
            self.dfn_info_list.append(dfn_info)


            fvs = verts[face][np.newaxis]
            norms = get_normal(fvs)
            norms = norms.float()[0]
            norms_v = geometry.mean_vertex_normals(verts.shape[0], face, norms.numpy())
            norms_v = torch.from_numpy(norms_v).float()
            assert not torch.isnan(norms_v).any()
            self.norms.append(norms)
            self.norms_v.append(norms_v)
            self.inputs_target_v.append(torch.cat([verts, norms_v], dim=-1))
            self.pix2face_indices.append(indices)
            self.pix2face_values.append(values)
    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        return idx

