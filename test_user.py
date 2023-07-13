import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from mymodel import latent_space
from FWH_args import test_parser
import myutils
from deformation_transfer import Mesh, Transfer, deformation_gradient
from data_loader import  load_data_mf_v3, load_data_ict_live
from ICT_model import ICT_model

import igl
import trimesh
import vedo
from vedo import Plotter
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
from cupyx.scipy.sparse.linalg import SuperLU
from torch_sparse import coalesce, transpose
import os
from copy import deepcopy

expression_bases = [
    'browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimple_L', 'mouthDimple_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthFunnel', 'mouthLeft', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L', 'mouthPress_R', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmile_L', 'mouthSmile_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'noseSneer_L', 'noseSneer_R'
] 

def deform_ict(z):
    # Deform the ict face by the ict blendshapes
    global face_model, vedo_face_model
    z[z > 1] = 1
    z[z < -1] = -1
    deformed_vertices = face_model.deform(z[:, :53])[0] # Select z_FACS
    vedo_face_model.points(deformed_vertices)

def render_img(mesh, renderer, img_normalizer, img_enc='cnn', img_path=''):
    if renderer is not None: # Render the image and save
        img, fragments = renderer.renderbatch([torch.from_numpy(mesh.vertices).float().to('cuda')], [torch.from_numpy(mesh.faces).float().to('cuda')], reverse=True)
        zbuf = fragments.zbuf[0, ..., 0].float()
        img = torch.cat((img, zbuf.unsqueeze(0).unsqueeze(-1)), dim=-1)
        img[img == -1] = img.amax(dim=(0, 1, 2))[-1]  * 2
        img = img_normalizer.normalize(img)
        if img_enc == 'cnn':
            img = img[..., :3]
        p2f = fragments.pix_to_face[0, ..., 0].cpu().float()
        mapping = torch.zeros((mesh.faces.shape[0], p2f.shape[0]**2))
        for i in range(mapping.shape[0]):
            mapping[i][p2f.reshape(-1) == i] = 1
            if mapping[i].max() != 0:
                mapping[i] /= torch.sum(mapping[i])
        mapping = mapping.float().to_sparse()
        np.save(img_path, img.cpu().numpy())
        return img
    else: # Load the image if needed
        if len(img_path) > 0:
            img = torch.from_numpy(np.load(img_path))
        else:
            img = None
    return img

def calc_new_mesh(args, normalizer, model, myfunc, mesh, z, operators, dfn_info, img=None):

    lu_solver, idxs, vals, rhs = operators
    cents = myutils.calc_cent(mesh)
    _, norms = myutils.calc_norm(mesh)
    cents = torch.from_numpy(cents).float().unsqueeze(0)
    norms = torch.from_numpy(norms).float().unsqueeze(0)
    inputs = torch.cat([cents, norms], dim=-1)
    inputs = inputs.to('cuda')
    norms_v = torch.from_numpy(igl.per_vertex_normals(mesh.vertices, mesh.faces)).float()

    if torch.isnan(norms_v).any():
        # If something wrong with the igl computation
        norms_v, _ = myutils.calc_norm(mesh)
        norms_v = torch.from_numpy(norms_v).float()
    input_source_v = torch.cat([torch.from_numpy(mesh.vertices), norms_v], dim=-1).float().unsqueeze(0)
    input_source_v = input_source_v.to('cuda')


    img_feat = model.img_feat(img) 

    inputs = torch.cat([inputs, img_feat.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)
    # if not args.img_only_mlp:
    input_source_v = torch.cat([input_source_v, img_feat.unsqueeze(1).expand(-1, input_source_v.shape[1], -1)], dim=-1)
           


    with torch.no_grad():
        model.update_precomputes(dfn_info)
        g_pred, z_iden = model.decode([inputs.float(), input_source_v.float()], z.float())
        g_pred = normalizer.inv_normalize(g_pred)
        g_pred = myutils.reconstruct_jacobians(g_pred, repr='matrix')
        out_pred = myfunc(g_pred, lu_solver, idxs, vals, rhs.shape)
        out_pred = out_pred - out_pred.mean(axis=[0, 1], keepdim=True)
        mesh_out = Mesh(out_pred[0].detach().cpu().numpy(), mesh.faces)
        theta, omega, A = myutils.calc_jacobian_stat(g_pred[0].detach().cpu().numpy())

    if z_iden is not None:
        z_iden = z_iden.detach().cpu().numpy()
    
    return mesh_out, theta, omega, A, z_iden


def generate_new_z(args, model, dataset, face, idx, dfn_info, calc_dfn_info=False, scale=1, shift=np.zeros((1, 3)), offset = np.zeros((1, 53))):
    # global dataset, face, args, latent_dim
    img_, f2v_t_idxs, f2v_t_vals, p2f_t_idxs, p2f_t_vals = None, None, None, None, None
    gradX, gradY = None, None
    if args.img:
        img_ = dataset.img[idx].to(args.device).unsqueeze(0)
        if args.pix2face:
            # Construct sparse matrices
            f2v_t_idxs = dataset.face2vertex_indices.to(args.device).unsqueeze(0)
            f2v_t_vals = dataset.face2vertex_values.to(args.device).unsqueeze(0)
            p2f_t_idxs = dataset.pix2face_indices[idx].to(args.device).unsqueeze(0)
            p2f_t_vals = dataset.pix2face_values[idx].to(args.device).unsqueeze(0)
    if dataset.gradX is not None:
        gradX = dataset.gradX[idx].unsqueeze(0).to(args.device)
        gradY = dataset.gradY[idx].unsqueeze(0).to(args.device)
        
    inputs_v = dataset.inputs_target_v[idx].to(args.device).unsqueeze(0)
    inputs_v_ = inputs_v.clone()
    inputs_v_[..., :3] = inputs_v_[..., :3] * scale + torch.from_numpy(shift).cuda()
    verts_ = inputs_v_[..., :3].squeeze()
       
    if type(dataset.face) == list: # Multiple topology in the same dataset
        face = dataset.face[idx].cpu().numpy()
        dfn_info = dataset.dfn_info_list[idx]
        dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
        
    mesh_expressed = Mesh(verts_.cpu().numpy(), face)

    with torch.no_grad():
        if calc_dfn_info:
            print('Recalculating dfn_info on-the-fly')
            mesh_input = Mesh(verts_.squeeze().cpu().numpy(), face)
            dfn_info = myutils.get_dfn_info(mesh_input)
            dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
        model.update_precomputes(dfn_info)
        if args.img:
            z = model.encode(inputs_v_, img_.to(inputs_v_.device), p2f_t_idxs=p2f_t_idxs, p2f_t_vals=p2f_t_vals, f2v_t_idxs=f2v_t_idxs, f2v_t_vals=f2v_t_vals, N_F=face.shape[0], batch_gradX=gradX, batch_gradY=gradY)
        else:
            z = model.encode(inputs_v_, batch_gradX=gradX, batch_gradY=gradY)
    z[..., :53] += torch.from_numpy(offset).cuda()
    return z, mesh_expressed

def get_mesh_operators(mesh):
    N_FACE = mesh.faces.shape[0]
    N_VERTEX = mesh.vertices.shape[0]
    transf =  Transfer(mesh, deepcopy(mesh))
    lu_solver = SuperLU(transf.lu)
    idxs, vals = coalesce(from_dlpack(transf.idxs.toDlpack()).long(), from_dlpack(transf.vals.toDlpack()), m=N_FACE *3, n=N_VERTEX)
    idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)
    rhs = transf.cupy_A.T
    return lu_solver, idxs, vals, rhs


def data_loading(args, normalizer, img_normalizer, wks_normalizer=None, ldmk_normalizer=None):
    idx_iden = args.iden_idx

    if args.dataset == 'ICT_live':
        iden_vec, latent_const, latent_const_val, latent_const_test, face, transfs, train_loaders, val_loaders, test_loaders,  dfn_info = load_data_ict_live(args.data_dir, args.data_head, 1, int(idx_iden) + 1, normalizer,  wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=True, use_f=True, use_source_v=True, use_landmarks=args.landmark, landmark_normalizer=ldmk_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
        if latent_const is not None:
            latent_const = torch.from_numpy(latent_const).float()
            latent_const_val = torch.from_numpy(latent_const_val).float()
        latent_const_test = torch.from_numpy(latent_const_test).float()
        dfn_info = dfn_info
        dataloader = test_loaders[idx_iden]
    elif args.dataset == 'MF':
        n_mf = int(idx_iden) + 1
        face, transfs, train_loaders, val_loaders, test_loaders, dfn_info = load_data_mf_v3(args.data_dir, args.data_head, 1, n_mf, normalizer,  wks_normalizer=wks_normalizer, feature_type=args.feature_type,  global_encoder=args.global_encoder, cache_dir=args.cache_dir, device=args.device, only_test=True, use_f=True, use_source_v=True, use_landmarks=args.landmark, landmark_normalizer=ldmk_normalizer, use_img=args.img, img_normalizer=img_normalizer, use_pix2face=args.pix2face, grad=args.grad, dfn_info_per_loader=args.dfn_info_per_loader)
        transfs_shift_mf = {'train': range(n_mf), 'val': range(n_mf), 'test': range(n_mf)}
        latent_const = None
        latent_const_test = None
        dataloader = test_loaders[idx_iden]

    return face, transfs, test_loaders, latent_const, latent_const_test, dataloader, dfn_info 
    

def model_loading(args, dfn_info):
    global_encoder_in_shape = 6 if args.feature_type == 'cents&norms' else 12
    in_shape = 6

    model = latent_space(global_encoder_in_shape, in_shape=in_shape,  out_shape=9, pre_computes=dfn_info, latent_shape=args.feat_dim, iden_blocks=args.iden_blocks, hid_shape=args.mlp_hid_channel,  residual=False, global_pn=args.global_pn, sampling=args.sampling, number_gn=args.num_gn, dfn_blocks=args.dfn_blocks, global_pn_shape=args.global_pn_shape, img_encoder=args.img_enc,  img_feat=args.img_feat, img_only_mlp=False, img_warp=args.img_warp)
    
    ckpt = torch.load(os.path.join(args.save_dir, args.load_head, args.load_head + f'_{args.resume_id}.pth'), map_location='cuda:0')
    model = myutils.load_state_dict(model, ckpt['model'])

    if model.global_pn is not None:
        model.global_pn.update_precomputes(dfn_info)
    model.float()
    model.to('cuda')
    return model

def load_mesh(mesh, renderer, scale, shift,  device='cuda', use_img=True, process=True, img_normalizer=None, img_path=''):
    mesh = deepcopy(mesh)
    
    if process:
        mesh = myutils.get_biggest_connected(mesh)
        mesh = myutils.remove_degenerated_triangles(mesh)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

    mesh.vertices *= scale
    mesh.vertices += shift
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh_operators = get_mesh_operators(mesh)
    mesh_dfn_info = myutils.get_dfn_info(mesh)
    mesh_dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in mesh_dfn_info]
    img  = render_img(mesh, renderer, img_normalizer, img_path=img_path)
    img = img.to(device)

    return mesh, mesh_operators, mesh_dfn_info, img


if __name__ == '__main__':

    np.warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore')
    plot = True
    compare = False
    process = True
    draw_color = False
    random = False
    blendshape_weights = True
    fix_z = False
    p = test_parser()
    args = p.parse_args()
    idx_iden = args.iden_idx
    recalculate_dfn_info = args.recalculate_dfn_info
    offset_str = args.offset
    offset_str = offset_str.split(',')
    offset = np.zeros((1, 53))
    for substr in offset_str:
        substr = substr.split(':')
        if len(substr) == 2:
            offset[0, int(substr[0])] = float(substr[1])
    


    save_dir = os.path.join(args.save_dir, args.load_head, args.load_head + f'_{args.resume_id}_test_inverse_rigging', os.path.basename(args.mesh_file))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scale = args.scale
    shift = np.array(args.shift)
    latent_dim = args.feat_dim

    img_path = args.mesh_file.replace('.obj', '_img.npy')
    if len(args.mesh_to_sample) > 0:
        mesh_to_sample_img_path = args.mesh_to_sample.replace('.obj', '_img.npy')


    # Initializing variables

    identities = sorted(os.listdir(os.path.join(args.data_dir, args.data_head)))
    identities = [_ for _ in identities if 'GHS' in _]

    normalizer = myutils.Normalizer(args.std_file, args.device)
    img_normalizer = myutils.Normalizer_img(args.img_file, args.device)
    renderer = myutils.renderer(view_d=2.5, img_size=256, fragments=True)

    # ICT loading
    face_model = ICT_model.load(type=1)
    vedo_face_model = vedo.Mesh([np.array(face_model.vertices), np.array(face_model.faces.reshape(-1, 3))])
    
    # Data loading

    face, transfs, test_loaders, latent_const, latent_const_test, dataloader, dfn_info = data_loading(args, normalizer, img_normalizer)
    dataset = dataloader.dataset
    
    # Mesh loading & preprocessing

    orig_mesh, orig_mesh_operators = None, None
    orig_img = None

    # Load the neural mesh of the dataset

    orig_mesh = Mesh(dataset.neutral_verts.numpy(), dataset.face.numpy())
    orig_mesh, orig_mesh_operators, dfn_info, orig_img = load_mesh(orig_mesh, renderer, 1, np.array([0, 0, 0]), process=False, img_normalizer=img_normalizer)
    orig_img = dataset.neutral_img.unsqueeze(0).to(args.device)
    data_mean = orig_mesh.vertices.mean(axis=0)
    dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    EPS = args.EPS

    
    # Load the mesh to be deformed

    mesh = trimesh.load(args.mesh_file, process=process, maintain_order=True)
    mesh, mesh_operators, mesh_dfn_info, img = load_mesh(mesh, renderer, scale, shift, img_normalizer=img_normalizer, img_path=img_path)
    N_FACE = mesh.faces.shape[0]
    N_VERTEX = mesh.vertices.shape[0]
    mesh_v = np.copy(mesh.vertices)
    global_scale = 1
    global_shift = np.zeros((1, 3))
    global_scale_source =1
    global_shift_source = np.zeros((1, 3))

    print(f'Mesh properties:\n\tVertices:\t{mesh.vertices.shape[0]}\n\tFaces:\t{mesh.faces.shape[0]}')

    # Model loading

    myfunc = deformation_gradient.apply
    model = model_loading(args, mesh_dfn_info)
    Loss = torch.nn.MSELoss()
    

    # Initialize the target information
    
    idx = 0
    z0 = torch.zeros((1, latent_dim)).to('cuda')
    z = z0
    exp = 0
    counter = 0

    # Calculating
    if orig_mesh is not None:
        mesh_expressed, _, _, _, _ = calc_new_mesh(args, normalizer, model, myfunc, orig_mesh, z0, orig_mesh_operators, dfn_info, img=orig_img) # Plot 0

    mesh_out, theta, omega, A, z_iden = calc_new_mesh(args, normalizer, model, myfunc, mesh, z0, mesh_operators, mesh_dfn_info, img=img) # Plot 1

    if plot:
        colormap = 'jet'
        cmap_linear = 'jet'
        if orig_mesh is not None:
            mesh_gt = vedo.Mesh([np.array(mesh_expressed.vertices), np.array(mesh_expressed.faces.reshape(-1, 3))]) # P 0
        else:
            mesh_gt = vedo.Mesh([np.array(dataset.verts[0]), np.array(dataset.face[0])])

        mesh_vedo = vedo.Mesh([np.array(mesh_out.vertices), np.array(mesh_out.faces.reshape(-1, 3))]) 
        mesh_vedo_1 = mesh_vedo.clone() # P 1
        if draw_color:
            mesh_vedo_1.cmap(colormap, theta, on="cells", vmin=-3.14/2, vmax=3.14 / 2).addScalarBar()
        if compare:
            vertex_diff = ((mesh_expressed.vertices - mesh_out.vertices) **2).mean(axis=-1)
            mesh_vedo_2 = mesh_vedo.clone().cmap(cmap_linear, vertex_diff, vmin=0, vmax=1e-3).addScalarBar()

        plt = Plotter(N=4)
        plt.show(mesh_gt, at=0)
        if compare:
            plt.show(mesh_vedo_2, at=2)
            plt.show(vedo.Text2D('||V - V_gt||^2', pos=(0.05, 0.95)), at=2)
        plt.show(mesh_vedo_1, at=1)
        idx_text = vedo.Text2D(f'{idx}', pos=(0.05, 0.85))
        plt.show(idx_text)
        plt.show(__doc__, at=0)
        plt.show(vedo.Text2D(f'Vertices: {N_VERTEX}', pos=(0.05, 0.95)), at=0)
        plt.show(vedo.Text2D(f'Faces: {N_FACE}', pos=(0.05, 0.90)), at=0)
        plt.show(mesh_gt, at=3)
        np.set_printoptions(precision=3)

    

        # The following are button and slider functions
        
        def make_slider_xyz(axis): # Shift the source/target mesh
            def slider(widget, event):
                global z0, z, mesh, mesh_v, global_shift, global_scale, z_iden, fix_z, global_scale_source, global_shift_source,  orig_mesh
                
                value = widget.GetRepresentation().GetValue()
                global_shift[0, axis] = value 
                mesh.vertices = mesh_v *  global_scale + global_shift

                if fix_z:
                    z_ = torch.cat([torch.from_numpy(z_iden).to(z.device), z], dim=-1)
                else:
                    z_ = z

                # Update the target based on the new z
                
                mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z_, mesh_operators, mesh_dfn_info, img=img)
                
                deform_ict(z)
                points = mesh_.vertices

                mesh_vedo_1.points(points)
                if draw_color:
                    mesh_vedo_1.cmap(colormap, theta, on="cells", vmin=-3.14/2, vmax=3.14 / 2)

                if compare:
                    vertex_diff = ((mesh_expressed.vertices - mesh_out.vertices) **2).mean(axis=-1)
                    mesh_vedo_2.points(points)
                    mesh_vedo_2.cmap(cmap_linear, vertex_diff, vmin=0, vmax=1e-3)
                
                print('mesh_in:', mesh.vertices.mean(axis=0))
                print('mesh_out:', points.mean(axis=0))
                    
                    

            return slider
        
        def make_slider_scale(): # Scale the source/target mesh
            def slider(widget, event):
                global z0, z, mesh, mesh_v, global_shift, global_scale, global_scale_source, global_shift_source, orig_mesh
                value = widget.GetRepresentation().GetValue()
                

                global_scale = value
                mesh.vertices = mesh_v * global_scale + global_shift

                if orig_mesh is not None: # Update the source mesh base on the control
                    mesh_gt.points(mesh_expressed.vertices)

                mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
                deform_ict(z)
                points = mesh_.vertices
                mesh_vedo_1.points(points)

            return slider
        
        def button_func_bound_transform():
            global z0, z, mesh, mesh_v, global_shift, global_scale, orig_mesh
            mesh2save = deepcopy(mesh)
            mesh2save.vertices = mesh_v * global_scale + global_shift
            mesh2save.export(args.mesh_file.replace('.obj', '_transformed.obj'))
            print(f"Apply transformation to {args.mesh_file.replace('.obj', '_transformed.obj')}")
 


        def apply_z_code(idx):
            global z, orig_mesh, mesh_gt, idx_text
            z, mesh_expressed = generate_new_z(args, model, dataset, face, idx, dfn_info, calc_dfn_info=recalculate_dfn_info, shift=global_shift_source, scale=global_scale_source, offset=offset)
            mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
            deform_ict(z)
            points = mesh_.vertices
            mesh_vedo_1.points(points)

            if draw_color:
                mesh_vedo_1.cmap(colormap, theta, on="cells", vmin=-3.14/2, vmax=3.14 / 2)
            idx_text.text(f'{idx}')
            if orig_mesh is not None:
                mesh_gt.points(mesh_expressed.vertices)
            else:
                plt.show(at=0)
                plt.remove(mesh_gt, render=True)
                mesh_gt = vedo.Mesh([np.array(mesh_expressed.vertices), np.array(mesh_expressed.faces.reshape(-1, 3))])
                plt.show(mesh_gt, at=0)

            print(f'Update expression: {idx}')
            expression_code = z[0].detach().cpu().numpy()
            if blendshape_weights:
                print(''.join([f'{idx}: {expression_bases[idx]}: {code_:.3f}\n' for idx, code_ in enumerate(expression_code[:53])]))
                print(''.join([f'{code_:.3f}, ' for code_ in expression_code]))


        
        def button_func(): # Pick new z code
            global z0, z, mesh_expressed, mesh_gt, random, idx
            idx = int(input(f'Please input expression idx: [0, {len(dataset)}]\n'))
            apply_z_code(idx)
            

        def button_func_next(): # Next z code
            global z0, z, mesh_expressed, mesh_gt, idx
            idx = (idx + 1) % len(dataset)
            apply_z_code(idx)

        def button_func_random(): # Randomly pick a z code
            global z0, z, mesh_expressed, mesh_gt, idx
            idx = np.random.randint(0, len(dataset))
            apply_z_code(idx)


        def button_func_iden(): # Next identity in dataloader
            global z0, z, mesh_expressed, mesh_gt, idx, dataset, idx_iden
            idx_iden = (idx_iden + 1) % len(test_loaders)
            dataloader = test_loaders[idx_iden]
            dataset = dataloader.dataset
            z, mesh_expressed = generate_new_z(args, model, dataset, face, idx, dfn_info, calc_dfn_info=recalculate_dfn_info, shift=global_shift_source, scale=global_scale_source, offset=offset)
            mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
            deform_ict(z)
            points = mesh_.vertices
            mesh_vedo_1.points(points)

            if draw_color:
                mesh_vedo_1.cmap(colormap, theta, on="cells", vmin=-3.14/2, vmax=3.14 / 2)

            mesh_gt.points(mesh_expressed.vertices)
            print(f'Update identity: {idx_iden}')
        
        def button_save(): # Save the current three meshes
            global mesh_gt, mesh_vedo_1, vedo_face_model, save_dir, idx, z, scale_diff, scale_diff_gt
            if not os.path.exists(os.path.join(save_dir, 'inverse_rigging')):
                os.makedirs(os.path.join(save_dir, 'inverse_rigging'))
            vedo.io.write(mesh_gt, os.path.join(save_dir, 'inverse_rigging', f'{args.dataset}_{idx_iden:02d}_{idx:04d}_target.obj'))
            vedo.io.write(mesh_vedo_1, os.path.join(save_dir, 'inverse_rigging', f'{args.dataset}_{idx_iden:02d}_{idx:04d}_source.obj'))
            vedo.io.write(vedo_face_model, os.path.join(save_dir, 'inverse_rigging', f'{args.dataset}_{idx_iden:02d}_{idx:04d}_ict.obj'))
            np.save(os.path.join(save_dir, f'{idx_iden:02d}_{idx:04d}_z.npy'), z[0].detach().cpu().numpy())
            print(f'Save to {save_dir}/inverse_rigging/{args.dataset}_{idx:04d}')


                    
        def slider_exp(widget, event): # Change the intensity of AU
            global z0, z, exp, counter
            value = widget.GetRepresentation().GetValue()

            z[0, exp] = value

            mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
            if orig_mesh is not None:
                mesh_expressed_, _, _, _, _ = calc_new_mesh(args, normalizer,model, myfunc, orig_mesh, z, orig_mesh_operators, dfn_info, img=orig_img)
            deform_ict(z)
            points = mesh_.vertices

            mesh_vedo.points(points)
            if orig_mesh is not None:
                mesh_gt.points(mesh_expressed_.vertices)

            mesh_vedo_1.points(points)
                



        def button_func_exp_idx():
            global z0, z, mesh_expressed, mesh_gt, exp
            exp = int(input('Please input expression idx to change:\n'))
            mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
            points = mesh_.vertices
            mesh_vedo_1.points(points)
            mesh_gt.points(mesh_expressed.vertices)
            # print(z)
        
        

        # Adding buttons and sliders to the plot

        xyz = ['x', 'y','z']
        for i in range(3):
            plt.at(2).addSlider2D(
                make_slider_xyz(i),
                xmin=-2.00,
                xmax=4.00,
                value=0.00,
                pos=[(0.02, 0.05 + 0.1 + i * 0.05), (0.25, 0.05 + 0.1 + i * 0.05)],
                title= f'{xyz[i]} shift'
                # title="color number",
            )
        plt.at(2).addSlider2D(
            make_slider_scale(),
            xmin=0.2,
            xmax=4.0,
            value=1.00,
            pos=[(0.02, 0.3), (0.25, 0.3)],
            title='scale'
            # title="color number",
        )

        plt.at(2).addSlider2D(
            slider_exp,
            xmin=0,
            xmax=2,
            value=0.00,
            pos=[(0.02, 0.4), (0.25, 0.4)],
            title='AU scale'
            # title="color number",
        )
        

        
        plt.at(2).addButton(
            button_func_exp_idx,
            pos=(0.8, 0.6),  # x,y fraction from bottom left corner
            states=['code_idx'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        )
        plt.at(2).addButton(
            button_func,
            pos=(0.1, 0.1),  # x,y fraction from bottom left corner
            states=['input'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        )
        plt.at(2).addButton(
            button_func_next,
            pos=(0.4, 0.1),  # x,y fraction from bottom left corner
            states=['next'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        
        )
        plt.at(2).addButton(
            button_func_random,
            pos=(0.7, 0.1),  # x,y fraction from bottom left corner
            states=['random'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        )
        plt.at(2).addButton(
            button_func_iden,
            pos=(0.4, 0.05),  # x,y fraction from bottom left corner
            states=['iden'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        )
        plt.at(2).addButton(
            button_func_bound_transform,
            pos=(0.7, 0.05),  # x,y fraction from bottom left corner
            states=['save_trans'],
            c=["g"],
            bc=['w'],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False
        )
        plt.interactive().close()
    
