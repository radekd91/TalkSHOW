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
from scripts.beat_subjects import subject2genderbeta, subject2genderbeta_consistent
from pathlib import Path
from scripts.demo_zero_face import get_vertices, showpose2smplxpose


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


def process_results(result_fname, smplx_model, rendertool, config, args):
    # SHOW_infer_path = Path("/is/cluster/work/rdanecek/for_kiran/habibie")
    SHOW_infer_path = args.out_folder
    wav_path = Path("/ps/scratch/shared_files_kchhatre/radek/gesticulation_audios_new")
    cur_wav_file = (wav_path / Path(result_fname).stem).with_suffix(".wav")
    if cur_wav_file.exists():
        cur_wav_file = str(cur_wav_file)
    else:
        backup_wav_path = Path("/ps/scratch/shared_files_kchhatre/radek/gesticulation_audios_test")
        cur_wav_file = (backup_wav_path / Path(result_fname).stem).with_suffix(".wav")
        if cur_wav_file.exists():
            cur_wav_file = str(cur_wav_file)
        else:
            cur_wav_file = None


    results = np.load(result_fname)
    # if cur_wav_file 
    # subject = "wayne"
    # gender = np.array("male", dtype='<U7')
    # gender_wayne, betas_wayne = subject2genderbeta(subject) 
    try:
        subject = Path(result_fname).name.split("_")[1]
        gender, betas = subject2genderbeta(subject)
        gender_wayne, betas_wayne = subject2genderbeta_consistent(subject)
    except KeyError:
        subject = "wayne"
        gender_wayne = np.array("male", dtype='<U7')
        gender_wayne, betas_wayne = subject2genderbeta_consistent(subject) 
        gender, betas = subject2genderbeta(subject)

    result_list = [ torch.from_numpy( results)]
    zero_face_results = torch.from_numpy( results).clone()
    zero_face_results[:, 0:3] = 0
    zero_face_results[:, 165:265] = 0

    result_list_no_face = [zero_face_results]
    vertices_list, pose = get_vertices(smplx_model, torch.from_numpy(betas)[None, ...], result_list, config.Data.pose.expression, require_pose=True)

    # vertices_list_zero_face, pose_wayne = get_vertices_zero_face(smplx_model, torch.from_numpy( betas_wayne).to(device)[None, ...], result_list_no_face, config.Data.pose.expression, require_pose=True)
    vertices_list_zero_face, pose_wayne = get_vertices(smplx_model, torch.from_numpy( betas_wayne)[None, ...], result_list_no_face, config.Data.pose.expression, require_pose=True)

    # result_list = [res.to('cpu') for res in result_list]
    dict = np.concatenate(result_list[:], axis=0)
    # file_name = 'visualise/video/' + config.Log.name + '/' + \
    #             cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
    output_folder = Path(SHOW_infer_path) / f"{Path(cur_wav_file).stem}"/ f"id_{args.id}"
    output_folder = Path(SHOW_infer_path) / f"{Path(cur_wav_file).stem}"/ f"id_{args.id}"
    smplx_npz = output_folder / f"{Path(cur_wav_file).stem}.npz"
    # smplx_npz_wayne = output_folder / f"{Path(cur_wav_file).stem}_wayne.npz"
    smplx_npz_wayne = output_folder / f"{Path(cur_wav_file).stem}_{subject}.npz"
    output_folder.mkdir(exist_ok=True, parents=True)

    rendertool._render_sequences_helper(str(smplx_npz.with_suffix(".mp4")), cur_wav_file, vertices_list, stand=args.stand, face=args.only_face, whole_body=args.whole_body, transcript=None)
    rendertool._render_sequences_helper(str(smplx_npz_wayne.with_suffix(".mp4")), cur_wav_file, vertices_list_zero_face, stand=args.stand, face=args.only_face, whole_body=args.whole_body, transcript=None)

        
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
