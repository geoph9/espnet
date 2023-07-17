import os
import sys
import glob
import argparse
import logging
from typing import Union, Optional

import torch
import torchaudio
import s3prl
import numpy as np
import torch.nn.functional as F

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.encoder import wav2vec2_encoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet.utils.cli_utils import get_commandline_args
from typeguard import check_argument_types, check_return_type
from tqdm import tqdm
from kaldiio import save_mat


class MatScpWriter(NpyScpWriter):
    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f"{key}.mat"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), value)
        self.fscp.write(f"{key} {p}\n")

        # Store the file path
        self.data[key] = str(p)


def extract(
    data_dir: str,
    download_dir: str,
    dump_dir: str,
    sampling_rate: int,
    seed: int,
    log_level: Optional[Union[int, str]] = "INFO"
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    assert check_argument_types(), "Invalid arguments"
    ljspeech_dir = os.path.join(download_dir, "LJSpeech-1.1", "wavs")
    if not os.path.isdir(ljspeech_dir):
        raise FileNotFoundError(f"Directory LJSpeech-1.1 should exist under the provided --{download_dir=}")
    out_scp_path = os.path.join(data_dir, "feats.scp")
    dump_npy_dir = os.path.join(dump_dir, "npy_feats")
    dump_mat_dir = os.path.join(dump_dir, "mat_feats")    
    os.makedirs(dump_npy_dir, exist_ok=True)
    os.makedirs(dump_mat_dir, exist_ok=True)
    wav_files_list = list(glob.glob(os.path.join(ljspeech_dir, "*.wav")))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # w2v2_model_path = wav2vec2_encoder.download_w2v(w2v2_url, download_dir)
    fr = S3prlFrontend(
        fs=sampling_rate, 
        frontend_conf={
            "upstream": "wav2vec2",
        },
        download_dir=download_dir,
    ).to(device)
    logging.info(f"Using S3prlFrontend model from {download_dir}")
    npy_writer = NpyScpWriter(dump_npy_dir, out_scp_path + ".npy")
    mat_writer = MatScpWriter(dump_mat_dir, out_scp_path)
    lengths_filepath = os.path.join(data_dir, ".lengths")
    if os.path.isfile(lengths_filepath):
        with open(lengths_filepath, "r") as f:
            n_files_processed = len(f.readlines())
        if n_files_processed == len(wav_files_list):
            logging.info(f"We are skipping the feats.scp calculation since .lengths already exists and contains {n_files_processed} files.")
            return
    lengths_file = open(lengths_filepath, "w", encoding="utf-8")
    # TODO: Just use batching...
    # Iterate through all wav files and create the scp entry and the npy file.
    for wav_file in tqdm(wav_files_list):  # we assume it's .wav
        key = os.path.basename(wav_file).split(".wav")[0]
        out_mat_path = os.path.join(dump_dir, "mat_feats", f"{key}.mat")
        if os.path.isfile(os.path.join(dump_npy_dir, f"{key}.npy")):
            npy_feats = np.load(os.path.join(dump_npy_dir, f"{key}.npy"))
            feats_lengths = npy_feats.shape[0]
        else:
            wav, _fs = torchaudio.load(wav_file)
            wav = torchaudio.functional.resample(wav, orig_freq=_fs, new_freq=sampling_rate).to(device)
            wav_lengths = torch.tensor([wav.shape[1]]).to(device)
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
                feats, feats_lengths = fr(wav, wav_lengths)
            npy_feats = feats.detach().cpu().squeeze(0).numpy()
            feats_lengths = feats_lengths.item()
        save_mat_from_npy(npy_feats, out_mat_path)
        npy_writer[key] = npy_feats
        mat_writer[key] = npy_feats
        lengths_file.write(f"{key} {feats_lengths}\n")
    npy_writer.close()
    lengths_file.close()


def save_mat_from_npy(npy_arr, mat_out_path):
    if os.path.isfile(mat_out_path):
        return
    save_mat(mat_out_path, npy_arr)


def main():
    parser = argparse.ArgumentParser(
        description="W2V2 Feature Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Where the feats.scp file will be saved."
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="Where the .npy files of the wav2vec features will be saved."
    )
    parser.add_argument(
        "--download_dir", 
        type=str,
        required=True,
        help="Where the model is/will be saved. Needs to also include the LJSpeech-1.1 directory."
    )
    #parser.add_argument(
    #    "--w2v2_url", 
    #    type=str, 
    #    default="https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt",
    #    help="The url of the wav2vec 2.0 model you want to download"
    #)
    parser.add_argument(
        "--sampling_rate", "--fs", "--sr",
        type=int,
        required=True,
        help="The sampling rate of the input audio files"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    print(get_commandline_args(), file=sys.stderr)
    args = parser.parse_args()
    kwargs = vars(args)
    extract(**kwargs)

if __name__ == "__main__":
    main()
