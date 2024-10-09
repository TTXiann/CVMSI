import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_transformer import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.encoder import ChangeDetector, AddSpatialInfo
from models.transformer_decoder import Speaker

from utils.utils import set_mode, load_checkpoint, decode_sequence, decode_sequence_transformer, coco_gen_format_save
from tqdm import tqdm

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--snapshot', type=int, required=True)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)


# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]

default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name

output_dir = os.path.join(exp_dir, exp_name)

test_output_dir = os.path.join(output_dir, 'test_output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

caption_output_path = os.path.join(test_output_dir, 'captions', 'test')
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)


snapshot_dir = os.path.join(output_dir, 'snapshots')
snapshot_file = 'checkpoint_%d.pt' % (args.snapshot)
snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
checkpoint = load_checkpoint(snapshot_full_path)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']


# Load modules
change_detector = ChangeDetector(cfg)
change_detector.load_state_dict(change_detector_state, strict=True)
change_detector = change_detector.to(device)

speaker = Speaker(cfg)
speaker.load_state_dict(speaker_state, strict=True)
speaker.to(device)

spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

# Data loading part
test_dataset, test_loader = create_dataset(cfg, 'test')
idx_to_word = test_dataset.get_idx_to_word()

set_mode('eval', [change_detector, speaker])
with torch.no_grad():
    test_iter_start_time = time.time()

    result_sents_pos = {}
    result_sents_neg = {}

    for i, batch in tqdm(enumerate(test_loader)):

        d_feats, sc_feats, nsc_feats , ids = batch
        d_feats, sc_feats, nsc_feats = d_feats.to(device), sc_feats.to(device), nsc_feats.to(device)

        d_feats, nsc_feats, sc_feats = spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats) 
        
        encoder_output_sc = change_detector(d_feats, sc_feats)
        encoder_output_nsc  = change_detector(d_feats, nsc_feats)

        speaker_output_sc, attention_weight_sc = speaker.sample(encoder_output_sc, sample_max=1)
        speaker_output_nse, attention_weight_nsc = speaker.sample(encoder_output_nsc, sample_max=1)

        gen_sents_sc = decode_sequence_transformer(idx_to_word, speaker_output_sc[:, 1:])
        gen_sents_nsc = decode_sequence_transformer(idx_to_word, speaker_output_nse[:, 1:])

        for j in range(len(gen_sents_sc)):
            sent_pos = gen_sents_sc[j]
            sent_neg = gen_sents_nsc[j]
            image_id = ids[j].split('_')[-1]
            result_sents_pos[image_id] = sent_pos
            result_sents_neg[image_id + '_n'] = sent_neg

   
    test_iter_end_time = time.time() - test_iter_start_time
    print('Test took %.4f seconds' % test_iter_end_time)

    result_save_path_pos = os.path.join(caption_output_path, 'sc_results.json')
    result_save_path_neg = os.path.join(caption_output_path, 'nsc_results.json')
    coco_gen_format_save(result_sents_pos, result_save_path_pos)
    coco_gen_format_save(result_sents_neg, result_save_path_neg)

    
    

