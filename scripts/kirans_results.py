import os, sys 
if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())
from pathlib import Path
import numpy as np
import torch 
import smplx
from visualise.rendering import RenderTool
import pickle

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


def smplxpose2showpose(pose): 
    pose = pose.view(-1)
    jaw_idx = slice(3*smplx_skeleton_inv['jaw'], 3*smplx_skeleton_inv['jaw'] + 3)
    jaw_pose = pose[jaw_idx][None, ...]

    leye_idx = slice(3*smplx_skeleton_inv['left_eye'], 3*smplx_skeleton_inv['left_eye'] + 3)
    reye_idx = slice(3*smplx_skeleton_inv['right_eye'], 3*smplx_skeleton_inv['right_eye'] + 3)
    leye_pose = pose[leye_idx][None, ...]
    reye_pose = pose[reye_idx][None, ...]

    global_orient_idx = slice(3*smplx_skeleton_inv['pelvis'], smplx_skeleton_inv['pelvis']+3)  # Global orientation is often the first three values in pose vector, please adjust if not
    global_orientation = pose[global_orient_idx][None, ...]

    body_pose_idx = slice(3*smplx_skeleton_inv['left_hip'], 3*smplx_skeleton_inv['right_wrist'] + 3)
    body_pose = pose[body_pose_idx][None, ...]

    left_hand_idx = slice(3*smplx_skeleton_inv['left_index1'], 3*smplx_skeleton_inv['left_thumb3'] + 3)
    left_hand = pose[left_hand_idx][None, ...]

    right_hand_idx = slice(3*smplx_skeleton_inv['right_index1'], 3*smplx_skeleton_inv['right_thumb3'] + 3)
    right_hand = pose[right_hand_idx][None, ...]
    full_body = np.concatenate(
            (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand, right_hand))
    return full_body

def showpose2smplxpose(full_body):
    """ Inverse of smplxpose2showpose
    """ 
    full_body = full_body.squeeze()  # remove the batch dimension
    pose = np.zeros((3*len(smplx_skeleton),))

    # Indices of body parts in full_body
    jaw_pose_idx = slice(0, 3)
    leye_pose_idx = slice(3, 6)
    reye_pose_idx = slice(6, 9)
    global_orient_idx = slice(9, 12)
    body_pose_idx = slice(12, 3*(smplx_skeleton_inv['right_wrist']+1))
    left_hand_idx = slice(body_pose_idx.stop, body_pose_idx.stop + 3*(smplx_skeleton_inv['left_thumb3']-smplx_skeleton_inv['left_index1']+1))
    right_hand_idx = slice(left_hand_idx.stop, left_hand_idx.stop + 3*(smplx_skeleton_inv['right_thumb3']-smplx_skeleton_inv['right_index1']+1))

    # Assign parts back into the pose vector
    pose[3*smplx_skeleton_inv['jaw']:3*smplx_skeleton_inv['jaw']+3] = full_body[jaw_pose_idx]
    pose[3*smplx_skeleton_inv['left_eye']:3*smplx_skeleton_inv['left_eye']+3] = full_body[leye_pose_idx]
    pose[3*smplx_skeleton_inv['right_eye']:3*smplx_skeleton_inv['right_eye']+3] = full_body[reye_pose_idx]
    pose[3*smplx_skeleton_inv['pelvis']:3*smplx_skeleton_inv['pelvis']+3] = full_body[global_orient_idx]
    pose[3*smplx_skeleton_inv['left_hip']:3*smplx_skeleton_inv['right_wrist']+3] = full_body[body_pose_idx]
    pose[3*smplx_skeleton_inv['left_index1']:3*smplx_skeleton_inv['left_thumb3']+3] = full_body[left_hand_idx]
    pose[3*smplx_skeleton_inv['right_index1']:3*smplx_skeleton_inv['right_thumb3']+3] = full_body[right_hand_idx]
    return pose

global_orient = torch.tensor([3.0747, -0.0158, -0.0152])

def render_npz(smplx_model, rendertool, npz_file, unmangle=True):
    if npz_file.suffix == '.npz':
        results = np.load(npz_file)
        betas = results.get('betas')
        poses = results.get('poses')
        # trans = results.get('trans')
    elif npz_file.suffix == '.pkl':
        with open(npz_file, 'rb') as f:
            results = pickle.load(f)
        betas = results.get('betas')[0]
        poses = results.get('jaw_pose')
        # trans = results.get('trans')
    else:
        raise ValueError('Unknown file type: {}'.format(npz_file.suffix))

    with open(Path('215481-00_01_36-00_01_39.pkl'), 'rb') as f: 
        r_ = pickle.load(f)
    betas = r_['betas'][0]

    # if betas too short, pad with zeros
    if betas.shape[0] < 300:
        betas = np.concatenate([betas, np.zeros(300 - betas.shape[0])])

    poses = torch.from_numpy(poses).double()
    betas = torch.from_numpy(betas).double()
    vertices_list = []
    vertices = []
    global_orient = torch.tensor([[[ 2.9450,  0.0739, -0.2503]]])
    for frame_idx in range(poses.shape[0]):
        pose_i = poses[frame_idx]
        expression = torch.zeros(1,100)
        if npz_file.suffix == '.pkl':
            jaw_pose =      torch.from_numpy(results['jaw_pose'][frame_idx:frame_idx+1]).double()
            leye_pose =     torch.from_numpy(results['leye_pose'][frame_idx:frame_idx+1]).double()
            reye_pose =     torch.from_numpy(results['reye_pose'][frame_idx:frame_idx+1]).double()
            # global_orient = torch.from_numpy(results['global_orient'][frame_idx:frame_idx+1]).double()
            body_pose =     torch.from_numpy(results['body_pose_axis'][frame_idx:frame_idx+1]).double()
            left_hand_pose =torch.from_numpy(results['left_hand_pose'][frame_idx:frame_idx+1]).double()
            right_hand_pose=torch.from_numpy(results['right_hand_pose'][frame_idx:frame_idx+1]).double()
            output = smplx_model(betas=betas[None, ...],
                                    expression=expression,
                                    jaw_pose=jaw_pose,
                                    leye_pose=leye_pose,
                                    reye_pose=reye_pose,
                                    global_orient=global_orient,
                                    body_pose=body_pose,
                                    left_hand_pose=left_hand_pose,
                                    right_hand_pose=right_hand_pose,
                                    return_verts=True)
        elif not unmangle:
            pose_i = pose_i.view(-1)[None, ...]

            output = smplx_model(betas=betas[None, ...],
                                    expression=expression,
                                    jaw_pose=pose_i[:, 0:3],
                                    leye_pose=pose_i[:, 3:6],
                                    reye_pose=pose_i[:,6:9],
                                    # global_orient=pose_i[:,9:12],
                                    global_orient=global_orient,
                                    body_pose=pose_i[:,12:75],
                                    left_hand_pose=pose_i[:,75:120],
                                    right_hand_pose=pose_i[:,120:165],
                                    return_verts=True)
        else: 
            pose_i = pose_i.view(-1)
            jaw_idx = slice(3*smplx_skeleton_inv['jaw'], 3*smplx_skeleton_inv['jaw'] + 3)
            jaw_pose = pose_i[jaw_idx][None, ...]

            leye_idx = slice(3*smplx_skeleton_inv['left_eye'], 3*smplx_skeleton_inv['left_eye'] + 3)
            reye_idx = slice(3*smplx_skeleton_inv['right_eye'], 3*smplx_skeleton_inv['right_eye'] + 3)
            leye_pose = pose_i[leye_idx][None, ...]
            reye_pose = pose_i[reye_idx][None, ...]

            global_orient_idx = slice(3*smplx_skeleton_inv['pelvis'], smplx_skeleton_inv['pelvis']+3)  # Global orientation is often the first three values in pose vector, please adjust if not
            global_orientation = pose_i[global_orient_idx][None, ...]

            body_pose_idx = slice(3*smplx_skeleton_inv['left_hip'], 3*smplx_skeleton_inv['right_wrist'] + 3)
            body_pose = pose_i[body_pose_idx][None, ...]

            left_hand_idx = slice(3*smplx_skeleton_inv['left_index1'], 3*smplx_skeleton_inv['left_thumb3'] + 3)
            left_hand = pose_i[left_hand_idx][None, ...]

            right_hand_idx = slice(3*smplx_skeleton_inv['right_index1'], 3*smplx_skeleton_inv['right_thumb3'] + 3)
            right_hand = pose_i[right_hand_idx][None, ...]

            output = smplx_model(betas=betas[None, ...],
                                expression=expression,
                                jaw_pose=jaw_pose,
                                leye_pose=leye_pose,
                                reye_pose=reye_pose,
                                # global_orient=global_orientation,
                                    global_orient=global_orient,
                                body_pose=body_pose,
                                left_hand_pose=left_hand,
                                right_hand_pose=right_hand,
                                return_verts=True)
            # jaw_idx = 22 
            # jaw_pose = pose_i[jaw_idx][None, ...]
            # left_eye_idx = 23 
            # right_eye_idx = 24
            # leye_pose = pose_i[left_eye_idx][None, ...]
            # reye_pose = pose_i[right_eye_idx][None, ...]
            # left_hand_indices = torch.arange(25, )
            # output = smplx_model(betas=betas[None, ...],
            #             expression=expression,
            #             jaw_pose=jaw_pose,
            #             leye_pose=leye_pose,
            #             reye_pose=reye_pose,
            #             global_orient=pose_i[:,9:12],
            #             body_pose=pose_i[:,12:75],
            #             left_hand_pose=pose_i[:,75:120],
            #             right_hand_pose=pose_i[:,120:165],
            #             return_verts=True)
        v = output.vertices.detach().cpu().numpy()
        # if npz_file.suffix != '.pkl':
        #     v[:, :, 1] = -v[:, :, 1]
        #     v[:, :, 2] = -v[:, :, 2]
 
        vertices.append(v.squeeze())
        # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
        pose = output.body_pose
        # poses.append(pose.detach().cpu())
    vertices = np.asarray(vertices)
    vertices_list.append(vertices)
    # poses = torch.cat(poses, dim=0)
    # poses_list.append(poses)

    output_path = Path("kirans_results")
    output_path.mkdir(exist_ok=True, parents=True)

    out_vid = str(output_path / (npz_file.stem + '.mp4'))
    rendertool._render_sequences_helper(out_vid, None, vertices_list, stand=False, face=False, whole_body=True, transcript=None)
    


def main(): 
    # kirans_result_path = Path("/ps/scratch/shared_files_kchhatre/radek/inferred_npzs")
    kirans_result_path = Path('/ps/scratch/shared_files_kchhatre/radek/chkpoints')
    # kirans_result_path = Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results')


    # find all the npz files in the directory
    npz_files = sorted(list(kirans_result_path.glob("*.npz")))
    # npz_files = [Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results/10_kieks_0_65_65_1/id_0/10_kieks_0_65_65_1_wayne.npz')]
    # npz_files = [Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results/1st-page/id_0/1st-page_wayne_nooffset.npz')]
    # npz_files = [Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results/1st-page/id_0/1st-page_wayne.npz')]
    # npz_files = [Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results/1st-page/id_0/1st-page_wayne_plus_offset.npz')]
    # npz_files = [Path('/is/cluster/work/rdanecek/for_kiran/SHOW_results/1st-page/id_0/1st-page_wayne_minus_offset.npz')]
    # npz_files = [Path('scott_seq_0_SQJHk7_motion_smplx.npz')] # Kiran's sample output 
    # npz_files = [Path('1_wayne_0_66_66.npz')]  # Kiran's sample input (by Giorgo)
    # npz_files = [Path('215481-00_01_36-00_01_39.pkl')]
    # npz_files = [Path('/ps/scratch/shared_files_kchhatre/radek/inferred_npzs/')]

    use_pca = npz_files[0].suffix == '.pkl'

    smplx_path = './visualise/'
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=use_pca,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        # gender='ne',
                        dtype=torch.float64, )
    smplx_model = smplx.create(**model_params)#.to(device)

    rendertool = RenderTool('visualise/video/' + 'kiran')
    for i, fname in enumerate(npz_files): 
        render_npz(smplx_model, rendertool, fname, unmangle=True)
        sys.exit()



if __name__ == "__main__":
    main()