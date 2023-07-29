import os
import sys
import os 
if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor
from glob import glob

import numpy as np
import json
import smplx as smpl

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool
from scripts.beat_subjects import subject2genderbeta
from pathlib import Path


global device
device = 'cpu'

smplx_skeleton = {
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3'
}
smplx_skeleton_inv = {v: k for k, v in smplx_skeleton.items()}

def showpose2smplxpose(full_body):
    """ Inverse of smplxpose2showpose
    """ 
    N = full_body.shape[0]
    full_body = full_body.reshape((N, -1))  # remove the batch dimension
    pose = np.zeros((N, 3*len(smplx_skeleton),))

    # Indices of body parts in full_body (the SHOW pose vector)
    jaw_pose_idx = slice(0, 3)
    leye_pose_idx = slice(3, 6)
    reye_pose_idx = slice(6, 9)
    global_orient_idx = slice(9, 12)
    body_pose_idx = slice(12, 75)
    left_hand_idx = slice(75,120)
    right_hand_idx = slice(120, 165)

    # Assign parts back into the pose vector
    pose[:, 3*smplx_skeleton_inv['jaw']:3*smplx_skeleton_inv['jaw']+3] = full_body[:, jaw_pose_idx]
    pose[:, 3*smplx_skeleton_inv['left_eye']:3*smplx_skeleton_inv['left_eye']+3] = full_body[:, leye_pose_idx]
    pose[:, 3*smplx_skeleton_inv['right_eye']:3*smplx_skeleton_inv['right_eye']+3] = full_body[:, reye_pose_idx]
    pose[:, 3*smplx_skeleton_inv['pelvis']:3*smplx_skeleton_inv['pelvis']+3] = full_body[:, global_orient_idx]
    pose[:, 3*smplx_skeleton_inv['left_hip']:3*smplx_skeleton_inv['right_wrist']+3] = full_body[:, body_pose_idx]
    pose[:, 3*smplx_skeleton_inv['left_index1']:3*smplx_skeleton_inv['left_thumb3']+3] = full_body[:, left_hand_idx]
    pose[:, 3*smplx_skeleton_inv['right_index1']:3*smplx_skeleton_inv['right_thumb3']+3] = full_body[:, right_hand_idx]
    return pose

def init_model(model_name, model_path, args, config):
    if model_name == 's2g_face':
        generator = s2g_face(
            args,
            config,
        )
    elif model_name == 's2g_body_vq':
        generator = s2g_body_vq(
            args,
            config,
        )
    elif model_name == 's2g_body_pixel':
        generator = s2g_body_pixel(
            args,
            config,
        )
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if model_name == 'smplx_S2G':
        generator.generator.load_state_dict(model_ckpt['generator']['generator'])

    elif 'generator' in list(model_ckpt.keys()):
        generator.load_state_dict(model_ckpt['generator'])
    else:
        model_ckpt = {'generator': model_ckpt}
        generator.load_state_dict(model_ckpt)

    return generator


def init_dataloader(data_root, speakers, args, config):
    if data_root.endswith('.csv'):
        raise NotImplementedError
    else:
        data_class = torch_data
    if 'smplx' in config.Model.model_name or 's2g' in config.Model.model_name:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='test',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            num_generate_length=config.Data.pose.generate_length,
            num_frames=30,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method,
            smplx=True,
            audio_sr=22000,
            convert_to_6d=config.Data.pose.convert_to_6d,
            expression=config.Data.pose.expression,
            config=config
        )
    else:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='val',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method
        )
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()
    infer_set = data_base.all_dataset
    infer_loader = data.DataLoader(data_base.all_dataset, batch_size=1, shuffle=False)

    return infer_set, infer_loader, norm_stats


def get_vertices(smplx_model, betas, result_list, exp, require_pose=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas,
                                 expression=i[j][165:265].unsqueeze_(dim=0) if exp else expression,
                                 jaw_pose=i[j][0:3].unsqueeze_(dim=0),
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
            pose = output.body_pose
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None
    

def get_vertices_zero_face(smplx_model, betas, result_list, exp, require_pose=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas,
                                 expression=torch.zeros_like(i[j][165:265].unsqueeze_(dim=0) if exp else expression),
                                 jaw_pose=torch.zeros_like(i[j][0:3].unsqueeze_(dim=0)),
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
            pose = output.body_pose
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(g_body, g_face, smplx_model, rendertool, config, args):
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
    num_sample = args.num_sample
    cur_wav_file = args.audio_file
    # SHOW_infer_path = Path("/is/cluster/work/rdanecek/for_kiran/SHOW_results")
    SHOW_infer_path = Path(args.out_folder)
    print("-------------------------------------")
    print(f"Processing results will be saved to {SHOW_infer_path}")
    print("-------------------------------------")
    id = args.id
    face = args.only_face
    stand = args.stand
    if face:
        body_static = torch.zeros([1, 162], device=device)
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)

    result_list = []
    result_list_no_face = []

    pred_face = g_face.infer_on_audio(cur_wav_file,
                                      initial_pose=None,
                                      norm_stats=None,
                                      w_pre=False,
                                      # id=id,
                                      frame=None,
                                      am=am,
                                      am_sr=am_sr
                                      )
    pred_face = torch.tensor(pred_face).squeeze().to(device)
    # pred_face = torch.zeros([gt.shape[0], 105])

    if config.Data.pose.convert_to_6d:
        pred_jaw = pred_face[:, :6].reshape(pred_face.shape[0], -1, 6)
        pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred_face.shape[0], -1)
        pred_face = pred_face[:, 6:]
    else:
        pred_jaw = pred_face[:, :3]
        pred_face = pred_face[:, 3:]

    id = torch.tensor([id], device=device)

    for i in range(num_sample):
        pred_res = g_body.infer_on_audio(cur_wav_file,
                                         initial_pose=None,
                                         norm_stats=None,
                                         txgfile=None,
                                         id=id,
                                         var=None,
                                         fps=30,
                                         w_pre=False
                                         )
        pred = torch.tensor(pred_res).squeeze().to(device)

        if pred.shape[0] < pred_face.shape[0]:
            repeat_frame = pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
            pred = torch.cat([pred, repeat_frame], dim=0)
        else:
            pred = pred[:pred_face.shape[0], :]

        body_or_face = False
        if pred.shape[1] < 275:
            body_or_face = True
        if config.Data.pose.convert_to_6d:
            pred = pred.reshape(pred.shape[0], -1, 6)
            pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
            pred = pred.reshape(pred.shape[0], -1)

        if config.Model.model_name == 's2g_LS3DCG':
            pred = torch.cat([pred[:, :3], pred[:, 103:], pred[:, 3:103]], dim=-1)
        else:
            pred = torch.cat([pred_jaw, pred, pred_face], dim=-1)

        # pred[:, 9:12] = global_orient
        pred = part2full(pred, stand)
        if face:
            pred = torch.cat([pred[:, :3], body_static.repeat(pred.shape[0], 1), pred[:, -100:]], dim=-1)
        pred_zero_face = pred.clone()
        pred_zero_face[:, 0:3] = 0
        pred_zero_face[:, 165:265] = 0
        # result_list[0] = poses2pred(result_list[0], stand)
        # if gt_0 is None:
        #     gt_0 = gt
        # pred = pred2poses(pred, gt_0)
        # result_list[0] = poses2poses(result_list[0], gt_0)

        result_list.append(pred)
        result_list_no_face.append(pred_zero_face)


    vertices_list, pose = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression, require_pose=True)
    subject = "wayne"
    gender = np.array("male", dtype='<U7')
    gender_wayne, betas_wayne = subject2genderbeta(subject) 
    # vertices_list_zero_face, pose_wayne = get_vertices_zero_face(smplx_model, torch.from_numpy( betas_wayne).to(device)[None, ...], result_list_no_face, config.Data.pose.expression, require_pose=True)
    vertices_list_zero_face, pose_wayne = get_vertices(smplx_model, torch.from_numpy( betas_wayne).to(device)[None, ...], result_list_no_face, config.Data.pose.expression, require_pose=True)

    # result_list = [res.to('cpu') for res in result_list]
    dict = np.concatenate(result_list[:], axis=0)
    # file_name = 'visualise/video/' + config.Log.name + '/' + \
    #             cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
    output_folder = Path(SHOW_infer_path) / f"{Path(cur_wav_file).stem}"/ f"id_{args.id}"
    output_folder = Path(SHOW_infer_path) / f"{Path(cur_wav_file).stem}"/ f"id_{args.id}"
    smplx_npz = output_folder / f"{Path(cur_wav_file).stem}.npz"
    smplx_npz_wayne = output_folder / f"{Path(cur_wav_file).stem}_wayne.npz"
    output_folder.mkdir(exist_ok=True, parents=True)

    rendertool._render_sequences_helper(str(smplx_npz.with_suffix(".mp4")), cur_wav_file, vertices_list, stand=stand, face=face, whole_body=args.whole_body, transcript=None)
    rendertool._render_sequences_helper(str(smplx_npz_wayne.with_suffix(".mp4")), cur_wav_file, vertices_list_zero_face, stand=stand, face=face, whole_body=args.whole_body, transcript=None)

        
    poses_ = dict[:, :165] 
    poses_normal_format = showpose2smplxpose(poses_) 
    # poses_normal_format += smplx_model.pose_mean[None, ...].cpu().numpy()
    # poses_normal_format -= smplx_model.pose_mean[None, ...].cpu().numpy()
    poses_normal_format = poses_normal_format.reshape(poses_.shape[0], -1, 3)
    jts2replacebyzero = [22, # include jaw pose 
                         1,2,3,4,5,7,8,10,11] # lower body
    # replace the joints with zero at jts2replacebyzero indices
    new_poses = poses_.copy()
    new_poses_normal_format = showpose2smplxpose(new_poses)
    new_poses_normal_format_minus_offset = new_poses_normal_format - smplx_model.pose_mean[None, ...].cpu().numpy()
    new_poses_normal_format_plus_offset = new_poses_normal_format + smplx_model.pose_mean[None, ...].cpu().numpy()
    new_poses_normal_format = new_poses_normal_format.reshape(new_poses_normal_format.shape[0], -1, 3)
    new_poses_normal_format_plus_offset = new_poses_normal_format_plus_offset.reshape(new_poses_normal_format_plus_offset.shape[0], -1, 3)
    new_poses_normal_format_minus_offset = new_poses_normal_format_minus_offset.reshape(new_poses_normal_format_minus_offset.shape[0], -1, 3)
    new_poses_normal_format[:, jts2replacebyzero, :] = 0
    new_poses_normal_format_minus_offset[:, jts2replacebyzero, :] = 0
    new_poses_normal_format_plus_offset[:, jts2replacebyzero, :] = 0

    trans = np.zeros((poses_.shape[0], 3))
    mocap_frame_rate_ = np.array(30, dtype='float64')

    np.savez(
        smplx_npz,
        poses=poses_normal_format,
        trans=trans,
        gender=gender, betas=betas,
        mocap_frame_rate=mocap_frame_rate_
    )

    np.savez(
        smplx_npz_wayne,
        poses=new_poses_normal_format,
        trans=trans,
        gender=gender_wayne, betas=betas_wayne,
        mocap_frame_rate=mocap_frame_rate_
    )
    # filename without extension
    fname = smplx_npz_wayne.parent / (smplx_npz_wayne.stem + "_minus_offset.npz")

    np.savez(
        fname,
        poses=new_poses_normal_format_minus_offset,
        trans=trans,
        gender=gender_wayne, betas=betas_wayne,
        mocap_frame_rate=mocap_frame_rate_
    )
    # add _plus_offset to the filename
    fname = smplx_npz_wayne.parent / (smplx_npz_wayne.stem + "_plus_offset.npz")
    np.savez(
        fname,
        poses=new_poses_normal_format_plus_offset,
        trans=trans,
        gender=gender_wayne, betas=betas_wayne,
        mocap_frame_rate=mocap_frame_rate_
    )
    # poses_normal_format += smplx_model.pose_mean[None, ...].cpu().numpy()
    
    
    print("results saved to ", smplx_npz)
    chmod_cmd = f"find {str(output_folder)} -print -type d -exec chmod 775 {{}} +"
    os.system(chmod_cmd)


def main():
    parser = parse_args()
    args = parser.parse_args()
    # device = torch.device(args.gpu)
    # torch.cuda.set_device(device)


    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    body_model_name = args.body_model_name
    body_model_path = args.body_model_path
    smplx_path = './visualise/'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = init_model(body_model_name, body_model_path, args, config)
    generator2 = None
    generator_face = init_model(face_model_name, face_model_path, args, config)

    print('init smlpx model...')
    dtype = torch.float64
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        # gender='ne',
                        dtype=dtype, )
    smplx_model = smpl.create(**model_params).to(device)
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)

    infer(generator, generator_face, smplx_model, rendertool, config, args)


if __name__ == '__main__':
    main()
