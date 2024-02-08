import torchaudio
import torchaudio.transforms as T
import torch
import utils
import os
import boto3

from repcodec.whisper_feature_reader import WhisperFeatureReader
from repcodec.RepCodec import RepCodec

def get_repcodec_model(checkpoint_path, device="cpu"):
    model = RepCodec(
        input_channels=1024,
        output_channels=1024,
        encode_channels=1024,
        decode_channels=1024,
        code_dim=1024,
        codebook_num=1,
        codebook_size=1024,
        bias=True,
        enc_ratios=(1, 1),
        dec_ratios=(1, 1),
        enc_strides=(1, 1),
        dec_strides=(1, 1),
        enc_kernel_size=3,
        dec_kernel_size=3,
        enc_block_dilations=(1, 1),
        enc_block_kernel_size=3,
        dec_block_dilations=(1, 1),
        dec_block_kernel_size=3
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict["model"]["generator"], strict=True)
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
repcodec_chkpt_path = ""

whisper_root, whisper_name, layer = "",  "medium", 24
reader = WhisperFeatureReader(whisper_root, whisper_name, layer, device=device)

repcodec_model = get_repcodec_model(repcodec_chkpt_path, device).to(device)
print("Repcodec model loaded")

s3 = boto3.resource("s3", 
                    region_name="auto",
                    endpoint_url="https://storage.googleapis.com",
                    aws_access_key_id=os.environ.get("HMAC_KEY"),
                    aws_secret_access_key=os.environ.get("HMAC_SECRET"))


def upload_object(key, filepath, bucket):
    s3.meta.client.upload_file(
        Filename=filepath,
        Bucket=bucket,
        Key=key,
    )


def process_one(save_folder=None, bucket_name=None, bucket_folder=None, filename=None):
    filename = filename.replace('.mp3','.wav').replace('.flac','.wav')
    cvec_path = os.path.join(save_folder, os.path.basename(filename) + ".cvec.pt")

    if os.path.exists(cvec_path):
        return

    wav, sr = torchaudio.load(filename)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)

    if sr == 16000:
        wav16k = wav
    else:
        wav16k = T.Resample(sr, 16000)(wav).to(device)

    whisper_feats = reader.get_feats_tensor(wav16k[0], rate=16000).unsqueeze(0).transpose(1, 2)

    with torch.no_grad():
        x = repcodec_model.encoder(whisper_feats)
        z = repcodec_model.projector(x)
    torch.save(z.cpu(), cvec_path)

    if bucket_folder is None or bucket_name is None:
        return
    
    key = os.path.join(bucket_folder, os.path.basename(save_folder), os.path.basename(cvec_path))

    upload_object(key, cvec_path, bucket_name)