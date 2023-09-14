import torch
from torch import autocast
from diffusers import DDIMScheduler
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid

###  추가된 부분 ㄱ
import torch.distributed as dist
import torch.multiprocessing as mp
###  ㄴ 추가된 부분 
MODEL_NAME = "checkpoints/CompVis/stable-diffusion-v1-4"
OUTPUT_DIR = "outputs/man-skiing"

## ==================================== 원래 코드대로 사용하려면 아랫부분 주석 해제 ====================================
## ============================================================================================================
# unet = UNet3DConditionModel.from_pretrained(OUTPUT_DIR, subfolder='unet', torch_dtype=torch.float16).to('cuda')
# scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder='scheduler')
# pipe = TuneAVideoPipeline.from_pretrained(MODEL_NAME, unet=unet, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_slicing()

# g_cuda = None

#@markdown Can set random seed here for reproducibility.
# g_cuda = torch.Generator(device='cuda')
# seed = 1234 #@param {type:"number"}
# g_cuda.manual_seed(seed)
## ============================================================================================================
## ============================================================================================================
#@markdown Run for generating videos.

prompt = "The area is submerged in water after the storm." #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
use_inv_latent = True #@param {type:"boolean"}
inv_latent_path = "outputs/man-skiing/inv_latents/ddim_latent-200.pt" #@param {type:"string"}
num_samples = 1 #@param {type:"number"}
guidance_scale = 12.5 #@param {type:"number"}
num_inference_steps = 200 #@param {type:"number"}
video_length = 16 #@param {type:"number"}
height = 256 #@param {type:"number"}
width = 256 #@param {type:"number"}

ddim_inv_latent = None
if use_inv_latent and inv_latent_path == "":
    from natsort import natsorted
    from glob import glob
    import os
    inv_latent_path = natsorted(glob(f"{OUTPUT_DIR}/inv_latents/*"))[-1]
    ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
    print(f"DDIM inversion latent loaded from {inv_latent_path}")


import os
import json


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()


def run_inference(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    unet = UNet3DConditionModel.from_pretrained(OUTPUT_DIR, subfolder='unet', torch_dtype=torch.float16).to('cuda')
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder='scheduler')
    pipe = TuneAVideoPipeline.from_pretrained(MODEL_NAME, unet=unet, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    g_cuda = None
    g_cuda = torch.Generator(device='cuda')
    # seed = 1234 #@param {type:"number"}
    # g_cuda.manual_seed(seed)

    with open('categoized_captions_1.json', 'r') as f:
        caption_dict = json.load(f)
    
    categories = ['volcano']
    for category_idx in range(0, len(categories)):
        category = categories[category_idx + rank]
        # print(category)
        # print(category_idx)
        # print(rank)
        category_dir = os.path.join("generated_outputs", category)
        video_dir = os.path.join(category_dir, "videos")
        caption_dir = os.path.join(category_dir, "captions")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(caption_dir, exist_ok=True)
        idx = 0
        for num in range(400): # range(생성할 비디오 수)
            prompt = caption_dict[category][idx]
            with autocast("cuda"), torch.inference_mode():
                videos = pipe(
                    prompt,
                    latents=ddim_inv_latent,
                    video_length=video_length,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    num_videos_per_prompt=num_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=g_cuda
                ).videos
            save_path = f"{video_dir}/{category}_{num}.gif"
            save_videos_grid(videos, save_path)
            caption_file = open(f"{caption_dir}/{category}_{num}.txt", "w")
            caption_file.write(prompt)
            caption_file.close()
            idx = (idx + 1) % len(caption_dict[category])


# with open('categoized_captions_1.json', 'r') as f:
#     caption_dict = json.load(f)
    
# categories = ['drought', 'earthquake', 'fire', 'flood']
# for category in categories:
#     category_dir = os.path.join("generated_outputs", category)
#     video_dir = os.path.join(category_dir, "videos")
#     caption_dir = os.path.join(category_dir, "captions")
#     os.makedirs(video_dir, exist_ok=True)
#     os.makedirs(caption_dir, exist_ok=True)
#     idx = 0
#     for num in range(400): # range(생성할 비디오 수)
#         prompt = caption_dict[category][idx]
#         with autocast("cuda"), torch.inference_mode():
#             videos = pipe(
#                 prompt,
#                 latents=ddim_inv_latent,
#                 video_length=video_length,
#                 height=height,
#                 width=width,
#                 negative_prompt=negative_prompt,
#                 num_videos_per_prompt=num_samples,
#                 num_inference_steps=num_inference_steps,
#                 guidance_scale=guidance_scale,
#                 generator=g_cuda
#             ).videos
#         save_path = f"{video_dir}/{category}_{num}.gif"
#         save_videos_grid(videos, save_path)
#         caption_file = open(f"{caption_dir}/{category}_{num}.txt", "w")
#         caption_file.write(prompt)
#         caption_file.close()
#         idx = (idx + 1) % len(caption_dict[category])


# with autocast("cuda"), torch.inference_mode():
#     videos = pipe(
#         prompt, 
#         latents=ddim_inv_latent,
#         video_length=video_length, 
#         height=height, 
#         width=width, 
#         negative_prompt=negative_prompt,
#         num_videos_per_prompt=num_samples,
#         num_inference_steps=num_inference_steps, 
#         guidance_scale=guidance_scale,
#         generator=g_cuda
#     ).videos

# save_dir = "./results" #@param {type:"string"}
# save_path = f"{save_dir}/{prompt}.gif"
# save_videos_grid(videos, save_path)

# # display
# from IPython.display import Image, display
# display(Image(filename=save_path))

#@markdown Free runtime memory
# exit()

def main():
    world_size = 2
    mp.spawn(
        run_inference,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()