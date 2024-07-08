from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import random
import string
from gtts import gTTS
import torch
from time import strftime
import os, sys
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
import warnings
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse

app = FastAPI()



save_dir = os.path.join("result_dir", strftime("%Y_%m_%d_%H.%M.%S"))
os.makedirs(save_dir, exist_ok=True)
device = "cuda"
batch_size = 8
current_code_path = ""
current_root_path = current_code_path
print(current_root_path)
os.environ['TORCH_HOME'] = os.path.join(current_root_path, 'checkpoints')

path_of_lm_croper = os.path.join(current_root_path,"checkpoints",  'shape_predictor_68_face_landmarks.dat')
path_of_net_recon_model = os.path.join(current_root_path,"checkpoints", 'epoch_20.pth')
dir_of_BFM_fitting = os.path.join(current_root_path,"checkpoints",  'BFM_Fitting')
wav2lip_checkpoint = os.path.join(current_root_path,"checkpoints", 'wav2lip.pth')

audio2pose_checkpoint = os.path.join(current_root_path,"checkpoints",  'auido2pose_00140-model.pth')
audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

audio2exp_checkpoint = os.path.join(current_root_path,"checkpoints",  'auido2exp_00300-model.pth')
audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

free_view_checkpoint = os.path.join(current_root_path,"checkpoints",  'facevid2vid_00189-model.pth.tar')

mapping_checkpoint = os.path.join(current_root_path,"checkpoints", 'mapping_00109-model.pth.tar')
facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')

# init model
print(path_of_net_recon_model)
preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

print(audio2pose_checkpoint)
print(audio2exp_checkpoint)
audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path,
                              wav2lip_checkpoint, device)

print(free_view_checkpoint)
print(mapping_checkpoint)
animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, device)

restorer_model = GFPGANer(model_path='checkpoints\\GFPGANv1.3.pth', upscale=1, arch='clean',
                          channel_multiplier=2, bg_upsampler=None)
enhancer_model = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                                  sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)


warnings.filterwarnings("ignore")


def main(source_video,driven_audio,enhancer,use_DAIN):
    pic_path = source_video
    audio_path = driven_audio
    enhancer_region = enhancer

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print(first_frame_dir)
    print(pic_path)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device)
    coeff_path = audio_to_coeff.generate(batch, save_dir)
    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, device)
    tmp_path, new_audio_path, return_path = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                                                        restorer_model, enhancer_model, enhancer_region)
    torch.cuda.empty_cache()
    if use_DAIN == True:
        import paddle
        from src.dain_model import dain_predictor
        paddle.enable_static()
        predictor_dian = dain_predictor.DAINPredictor(os.path.join(current_root_path,"dian_output"), weight_path="./checkpoints/DAIN_weight",
                                                      time_step=0.5,
                                                      remove_duplicates=False)
        frames_path, temp_video_path = predictor_dian.run(tmp_path)
        paddle.disable_static()
        save_path = return_path[:-4] + '_dain.mp4'
        command = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_video_path, new_audio_path, save_path)
        os.system(command)
    os.remove(tmp_path)
    return return_path




def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), text: str = Form(...)):
    file_contents = await file.read()
    video_file = file.filename
    video_extension = video_file.split(".")[-1]
    video_name = f"{generate_random_string(10)}.{video_extension}"
    video_path = f"temp/{video_name}"
    with open(video_path, "wb") as video:
        video.write(file_contents)
    mp3_name = f"{generate_random_string(10)}.mp3"
    mp3_path = f"temp/{mp3_name}"
    tts = gTTS(text, lang='en')
    tts.save(mp3_path)
    enhancer = "lip,face"
    result = main(video_path,mp3_path,enhancer,use_DAIN=False)


    # return {"filename": file.filename, "text": text}
    return FileResponse(result)
