import configargparse as argparse
import sys

def parser():
    p = argparse.ArgParser()
    p.add_argument('-c', '--config', required=True, is_config_file=True, help='config file path')

    p.add_argument('--EPS', type=float, default=1e-6)
    p.add_argument('--data_dir', type=str)
    p.add_argument('--data_head', type=str)
    p.add_argument('--save_dir', default='/media/qindafei/SSD/face/')
    p.add_argument('--save_head', default='FWH_pn_yes_feat_yes_all')
    p.add_argument('--pn_channel', type=int, default=40)
    p.add_argument('--mlp_hid_channel', type=int, default=128)
    p.add_argument('--do_norm', default=False, action='store_true')
    p.add_argument('--use_feat', default=False, action='store_true')
    p.add_argument('--feat_dim', type=int, default=5)
    p.add_argument('--global_encoder', type=str, default='pn')
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--use_gn', default=False, action='store_true')

    p.add_argument('--use_zero_embd_as_neutral', default=False, action='store_true')
    p.add_argument('--dataset', type=str)

    p.add_argument('--feature_type', default='cents&norms', choices = ['cents&norms', 'jacobians'])

    p.add_argument('--device', default='cuda:0')

    p.add_argument('--wks_std_file')
    p.add_argument('--mask', action='store_true')

    p.add_argument('--global_pn', default=False, action='store_true')
    p.add_argument('--sampling', default=0, type=int)
    p.add_argument('--pn_norm', default=False, action='store_true')
    p.add_argument('--zero_mean', default=False, action='store_true')
    p.add_argument('--std_file')
    p.add_argument('--save_exps', default=False, action='store_true')
    p.add_argument('--global_pn_shape', default=None, type=int)
    p.add_argument('--dfn_blocks', type=int, default=4)
    p.add_argument('--num_gn', type=int, default=64)
    p.add_argument('--iden_blocks', type=int, default=4)
    p.add_argument('--landmark', default=False, action='store_true')
    p.add_argument('--landmark_file')
    p.add_argument('--img', default=False, action='store_true')
    p.add_argument('--img_enc', default='none', choices=['cnn', 'unet', 'none'])
    p.add_argument('--img_file')
    p.add_argument('--pix2face', default=False, action='store_true')
    p.add_argument('--img_norm', default='bn')
    p.add_argument('--img_feat', default=32, type=int)
    p.add_argument('--img_only_mlp', default=False, action='store_true')
    p.add_argument('--img_warp', default=False, action='store_true')
    p.add_argument('--grad', default=False, action='store_true')
    p.add_argument('--dfn_info_per_loader', default=False, action='store_true')
    p.add_argument('--no_global_encoder_grad', default=False, action='store_true')
    p.add_argument('--no_iden_encoder_grad', default=False, action='store_true')
    p.add_argument('--per_sample_dfn_info', default=False, action='store_true')
    return p



def test_parser():
    p = parser()
    p.add_argument('--mesh_file')
    p.add_argument('--mesh_to_sample', default='')
    p.add_argument('--resume_id', default=0)
    p.add_argument('--shift', type=float, action='append')
    p.add_argument('--iden_idx', type=int, default=0)
    # p.add_argument('--test_type', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--range', type=int, action='append', default=[])
    p.add_argument('--load_head', type=str)
    p.add_argument('--scale', type=float, default=1)
    p.add_argument('--recalculate_dfn_info', '-r', action='store_true')
    p.add_argument('--offset', type=str, default='')
    return p