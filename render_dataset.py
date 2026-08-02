import numpy as np
import pickle
import os
from myutils.myutils import Mesh, renderer
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from myutils.myutils import renderer
import torch
from PIL import Image
import scipy
from torch.nn.utils.rnn import pad_sequence
import argparse
class ds(torch.utils.data.Dataset):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces

    def __len__(self):
        return len(self.verts)
    def __getitem__(self, idx):
        return self.verts[idx], self.faces

def render_dataset(dataset_path, renderer):

    files = sorted(glob(os.path.join(dataset_path, '*', '*.pkl')))
    for idx, f in enumerate(tqdm(files)):
        datatype = f.split('/')[-2]

        # Neutral. Three naming conventions are tried in order:
        #   <name>_neutral.obj : most datasets (ICT_live_100, ICT_single_coarse, ...)
        #   <name>.obj         : datasets whose neutral keeps the mesh name (ICT_holes, ...)
        #   <id>_neutral.obj   : sequences, where all clips of an identity share one neutral
        base = os.path.basename(f)
        for cand in (base.replace('_processed_matrix.pkl', '_neutral.obj'),
                     base.replace('_processed_matrix.pkl', '.obj'),
                     base.split('_')[0] + '_neutral.obj'):
            try:
                neutral_obj = Mesh.load(os.path.join(dataset_path, cand))
                break
            except:
                continue
        else:
            raise FileNotFoundError(f'No neutral mesh found for {f}')

        neutral_obj.vertices = neutral_obj.vertices.astype(np.float32)
        img_neutral, fragments = renderer.renderbatch([torch.from_numpy(neutral_obj.vertices).to('cuda')], [torch.from_numpy(neutral_obj.faces).to('cuda')], reverse=True)
        img_neutral = img_neutral[0].cpu().numpy().astype(np.float32)
        p2f = fragments.pix_to_face[0, ..., 0].cpu().float()
        zbuf = fragments.zbuf[0, ..., 0].cpu().float()
        mapping = torch.zeros((neutral_obj.faces.shape[0], p2f.shape[0]**2))
        for i in range(mapping.shape[0]):
            mapping[i][p2f.reshape(-1) == i] = 1
            if mapping[i].max() != 0:
                mapping[i] /= torch.sum(mapping[i])
        mapping = mapping.float().to_sparse()
        torch.save({'mapping': mapping, 'zbuf': zbuf}, os.path.join(dataset_path, os.path.basename(f).replace('_processed_matrix.pkl', '_neutral_img.pt')))
        
        np.save(os.path.join(dataset_path, os.path.basename(f).replace('_processed_matrix.pkl', '_neutral_img.npy')),img_neutral)
        img_neutral = (img_neutral * 255).astype(np.uint8)
        Image.fromarray(img_neutral).save(os.path.join(dataset_path, os.path.basename(f).replace('_processed_matrix.pkl', '_neutral_img.png')))

        # Data
        try:
            with open(f, 'rb') as ff:
                face, verts, gradients= pickle.load(ff).values()
        except ValueError:
            with open(f, 'rb') as ff:
                face, verts, gradients, _= pickle.load(ff).values()

        verts= verts.astype(np.float32)
        dataset = ds(verts, face)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
        imgs_all = []
        zbuf_all = []
        i_all = []
        v_all = []
        for v, face in dataloader:

            v = v.to('cuda')
            face = face.to('cuda')
            imgs, fragments = renderer.renderbatch(v, face, reverse=True)
            imgs = imgs.cpu().float().numpy().astype(np.float32)
            imgs_all.append(imgs)
            p2f = fragments.pix_to_face[..., 0].cpu()


            # t = time.time()
            for b in range(p2f.shape[0]):
                p2f_ = p2f[b].flatten()

                mapping = scipy.sparse.coo_matrix((np.ones(len(p2f_[p2f_!=-1])), ((p2f_[p2f_!=-1] - b * face.shape[1]).cpu().numpy(), torch.where(p2f_ != -1)[0].cpu().numpy())), shape=(face.shape[1], p2f.shape[1]**2))
                coo = mapping.multiply(1 / (mapping.sum(axis=1) + 1e-6))
                values = coo.data
                indices = np.vstack((coo.row, coo.col))

                i = torch.LongTensor(indices).transpose(0, 1)
                v = torch.FloatTensor(values).unsqueeze(1)
                i_all.append(i)
                v_all.append(v)

            z_buf = fragments.zbuf[..., 0].cpu().float()
            zbuf_all.append(z_buf)


            if not os.path.exists(os.path.join(dataset_path, 'rendered')):
                os.makedirs(os.path.join(dataset_path, 'rendered'))
            _ = np.random.randint(len(imgs))
            im = imgs[_]
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(os.path.join(dataset_path, 'rendered', os.path.basename(f).replace('_processed_matrix.pkl', f'_{_:03d}.png')))
        imgs_all = np.concatenate(imgs_all, axis=0)
        np.save(os.path.join(dataset_path, datatype, os.path.basename(f).replace('_processed_matrix.pkl', '_imgs.npy' )), imgs_all)
        zbuf_all = torch.cat(zbuf_all, dim=0)
        i_all = pad_sequence(i_all, batch_first=True).transpose(1, 2)
        v_all = pad_sequence(v_all, batch_first=True).squeeze()
        torch.save({'indices': i_all, 'values': v_all, 'zbuf':zbuf_all}, os.path.join(dataset_path, datatype, os.path.basename(f).replace('_processed_matrix.pkl', '_imgs.pt' )))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    datasets = args.dataset
    # datapaths = [os.path.join(prefix, _) for _ in datasets]
    renderer = renderer(view_d=2.5, img_size=256, fragments=True)
    # for datapath in datapaths:
    render_dataset(datasets, renderer)

