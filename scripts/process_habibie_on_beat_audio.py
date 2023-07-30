import os, sys
sys.path.append(os.getcwd())
from scripts.process_habibie import *
import math

def main():
    parser = parse_args()
    args = parser.parse_args()
    # device = torch.device(args.gpu)
    # torch.cuda.set_device(device)

    args.config_file = "./config/body_pixel.json"
    config = load_JsonConfig(args.config_file)

    # face_model_name = args.face_model_name
    # face_model_path = args.face_model_path
    # body_model_name = args.body_model_name
    # body_model_path = args.body_model_path
    smplx_path = './visualise/'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    # print('init model...')
    # generator = init_model(body_model_name, body_model_path, args, config)
    # generator2 = None
    # generator_face = init_model(face_model_name, face_model_path, args, config)

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

    args.audio_file = "/ps/scratch/shared_files_kchhatre/radek/habibie_et_al_inferred"
    path_to_audios = Path(args.audio_file)

    npy_files = sorted([str(p) for p in path_to_audios.glob('*.npy')])
    
    audios_per_shard = args.audios_per_shard
    shard_idx = args.shard_idx

    assert audios_per_shard > 0 
    assert shard_idx >= 0

    start_idx = audios_per_shard * shard_idx
    end_idx = min(audios_per_shard * (shard_idx + 1), len(npy_files))

    if start_idx >= end_idx:
        print('shard_idx is too large. Maximum num shard is {} when using shard size of {}'.format(int(math.ceil(len(npy_files) / audios_per_shard))), audios_per_shard)
        return
    print(f"Process shard {shard_idx} out of {int(math.ceil(len(npy_files) / audios_per_shard))}")

    for i, wav in enumerate(npy_files[start_idx:end_idx]):
        args.audio_file = wav
        # for id in range(4):
        #     args.id = id
        args.id = 0
        with torch.no_grad():
            process_results(wav, smplx_model, rendertool, config, args)


if __name__ == '__main__':
    main()
