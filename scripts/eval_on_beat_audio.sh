#!/bin/bash
source /home/rdanecek/.bashrc
source /home/rdanecek/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate talkshowv2
export HF_HOME=/ps/scratch/rdanecek/huggingface_cache
# /is/cluster/fast/rdanecek/envs/talkshowv2/bin/python scripts/eval_on_beat_audio.py --config_file ./config/body_pixel.json --infer --audio_file /ps/scratch/shared_files_kchhatre/radek/gesticulation_audios --id 0 --whole_body --shard_idx $1 --audios_per_shard $2
# /is/cluster/fast/rdanecek/envs/talkshowv2/bin/python  scripts/eval_on_beat_audio.py  --config_file $3 --infer --audio_file /ps/scratch/shared_files_kchhatre/radek/gesticulation_audios_new --id 0 --whole_body --shard_idx $1 --audios_per_shard $2 --out_folder $4
# /is/cluster/fast/rdanecek/envs/talkshowv2/bin/python  scripts/eval_on_beat_audio.py  --config_file $3 --infer --audio_file /ps/scratch/shared_files_kchhatre/radek/gesticulation_audios_test
/is/cluster/fast/rdanecek/envs/talkshowv2/bin/python  scripts/eval_on_beat_audio.py  --config_file $3 --infer --audio_file $4