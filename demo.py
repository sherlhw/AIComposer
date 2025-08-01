import torch
from PIL import Image
import numpy as np
import os
import cv2
import argparse
import random
import json
from tqdm import tqdm
import torch.nn as nn

from diffusers import DPMSolverMultistepScheduler, ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from model.sdxl import sdxl
from model.sdxl_inversion import Inversion
from model.sdxl_utils.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from model.sdxl_utils.modeling_clip import CLIPTextInvModel, CLIPTextInvModelWithProjection

from model.sdxl_utils.attention_processor import IPAttnProcessor2_0, CNAttnProcessor2_0, AttnProcessor2_0

class MLPCLIPFeatureNetwork(nn.Module):
    def __init__(self, input_dim=2*2048*8, hidden_dim1=128, output_dim=2048*8):
        super(MLPCLIPFeatureNetwork, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim),
        )
        
    def forward(self, clip_feature1, clip_feature2):
        batch_size, seq_len, feature_num = clip_feature1.shape
        concatenated_features = torch.cat((clip_feature1, clip_feature2), dim=2)
        flattened_features = concatenated_features.view(batch_size, -1)
        output_feature = self.mlp(flattened_features)
        output_feature = output_feature.view(batch_size, seq_len, feature_num)
        return output_feature

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size

    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def seed_everything(seed = None, workers = False):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        rank_zero_warn(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

def init_model(model_path, model_dtype="fp16", num_ddim_steps=50, load_controlnet=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if model_dtype == "fp16":
        torch_dtype = torch.float16
    elif model_dtype == "fp32":
        torch_dtype = torch.float32
    image_encoder_path = "./IP-Adapter/sdxl_models/image_encoder"
    if load_controlnet:
        # controlnet
        controlnet_path = "./ControlNet/controlnet-canny-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch_dtype).to(device)
        pipe = sdxl.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch_dtype, use_safetensors=True, variant=model_dtype)
    else:
        pipe = sdxl.from_pretrained(model_path, torch_dtype=torch_dtype, use_safetensors=True, variant=model_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
    pipe.to(device)
    text_encoder_inv = CLIPTextInvModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2_inv = CLIPTextInvModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    inversion = Inversion(pipe, num_ddim_steps, scheduler_type="DPMSolver", text_encoder_inv=text_encoder_inv, text_encoder_2_inv=text_encoder_2_inv)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device, dtype=torch_dtype)
    clip_image_processor = CLIPImageProcessor()
    # image proj model
    image_proj_model = ImageProjModel(
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(device, dtype=torch_dtype)
    return pipe, inversion, text_encoder_inv, text_encoder_2_inv, clip_image_processor, image_encoder, image_proj_model

def load_ip_adapter(pipe, image_proj_model):
    ip_ckpt = "./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
    state_dict = torch.load(ip_ckpt, map_location="cpu")
    image_proj_model.load_state_dict(state_dict["image_proj"])
    ip_layers = torch.nn.ModuleList(pipe.unet.attn_processors.values())
    ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
    return pipe, image_proj_model

def set_ip_adapter(pipe, num_tokens=4, device=None, torch_dtype=torch.float16):
    unet = pipe.unet
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0()
        else:
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_tokens
            ).to(device, dtype=torch_dtype)
    unet.set_attn_processor(attn_procs)
    if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
        if isinstance(pipe.controlnet, MultiControlNetModel):
            for controlnet in pipe.controlnet.nets:
                controlnet.set_attn_processor(CNAttnProcessor2_0(num_tokens=num_tokens))
        else:
            pipe.controlnet.set_attn_processor(CNAttnProcessor2_0(num_tokens=num_tokens))
    return pipe

def image2latent(pipe, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        else:
            if image.ndim==3:
                image=np.expand_dims(image,0)
            image = torch.from_numpy(image).to(torch.float16).to('cuda') / 127.5 - 1
            image = image.permute(0, 3, 1, 2)
            latents=[]
            for i,_ in enumerate(image):
                latent=pipe.vae.encode(image[i:i+1])['latent_dist'].mean
                latents.append(latent)
            latents=torch.stack(latents).squeeze(1)
            latents = latents * pipe.vae.config.scaling_factor
    return latents

def get_image_embeds(pil_image=None, clip_image_embeds=None, content_prompt_embeds=None, device=None, image_encoder=None, clip_image_processor=None, image_proj_model=None):
    if pil_image is not None:
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds =image_encoder(clip_image.to(device, dtype=torch.float16)).image_embeds
    else:
        clip_image_embeds = clip_image_embeds.to(device, dtype=torch.float16)
    
    if content_prompt_embeds is not None:
        clip_image_embeds = clip_image_embeds - content_prompt_embeds

    image_prompt_embeds = image_proj_model(clip_image_embeds)
    uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(clip_image_embeds))
    return image_prompt_embeds, uncond_image_prompt_embeds

def random_color():
    """生成随机颜色"""
    return [random.randint(0, 255) for _ in range(3)]

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_json', type=str, default='./examples/examples.json',
                        help='path to the input json')
    parser.add_argument('--save_path', type=str, default='./output/',
                        help='path to the output image folder')
    parser.add_argument('--no_prompt', action='store_true',
                        help='是否有文本prompt引导')
    parser.add_argument('--load_controlnet', action='store_true',
                        help='是否加载controlnet')
    parser.add_argument('--threshold_value', type=int, default=10,
                        help='反演步骤开始值，注入prompt的时间步阈值')
    parser.add_argument('--threshold_end', type=int, default=16,
                        help='注入prompt的时间步阈值')
    parser.add_argument('--use_adain_timestep_begin', type=int, default=10,
                        help='')
    parser.add_argument('--use_adain_timestep_end', type=int, default=14,
                        help='')
    parser.add_argument('--backgroud_maintain', type=int, default=20,
                        help='')
    parser.add_argument('--mlp_path', type=str, default='./mlp_model/mlp_sdxl.pth',
                        help='path to the mlp model')                    

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    save_path = args.save_path
    torch_dtype = torch.float16
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ddim_steps = 20

    base_model_path = "./stable-diffusion-xl-base-1.0"
    pipe, inversion, text_encoder_inv, text_encoder_2_inv, clip_image_processor, image_encoder, image_proj_model = init_model(
        base_model_path, 
        num_ddim_steps=ddim_steps, 
        load_controlnet=args.load_controlnet
    )
    pipe = set_ip_adapter(pipe, device=pipe.device)
    pipe, image_proj_model = load_ip_adapter(pipe, image_proj_model)
    seed = 3407
    seed_everything(seed)

    mlp_model_path = args.mlp_path
    hidden_dim = 512
    mlp_model = MLPCLIPFeatureNetwork(hidden_dim1=hidden_dim).to(device)
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
    mlp_model.eval()
        
    with open(args.input_json, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    os.makedirs(save_path, exist_ok=True)

    for item in data:
        background_path = item['background_image']
        background_mask_path = item['background_mask']

        if args.no_prompt:
            prompt = ''
        else:
            prompt = item['prompt']

        init_image = Image.open(background_path).convert("RGB")
        mask_image = Image.open(background_mask_path).convert("L")

        # 成比例resize到最大边为1024，且保证可以被8整除
        orig_w, orig_h = init_image.size
        image_scale = orig_w / orig_h
        min_l = max(orig_w, orig_h)
        if min_l == orig_w:
            W = int(1024)
            H = int((W // image_scale // 8) * 8)
        else:
            H = int(1024)
            W = int(int(H * image_scale) - (int(H * image_scale) % 8))
        init_image = init_image.resize((W, H), Image.LANCZOS)
        mask_image = mask_image.resize((W, H), Image.LANCZOS)
        # 获取mask的bbox
        image_with_mask = np.array(mask_image)
        coor = np.nonzero(image_with_mask)
        xmin = coor[0][0]
        xmax = coor[0][-1]
        coor[1].sort()
        ymin = coor[1][0]
        ymax = coor[1][-1]
        mask_w = ymax - ymin
        mask_h = xmax - xmin
        mask_image = mask_image.convert("RGB")

        # 加载参考图及其mask
        image_prompt_path = item['ref_image']
        image_prompt_mask_path = item['ref_mask']
        image_prompt = Image.open(image_prompt_path).convert("RGB")
        image_prompt_mask = Image.open(image_prompt_mask_path).convert("L")
        if image_prompt_mask.size != image_prompt.size:
            image_prompt_mask = image_prompt_mask.resize(image_prompt.size, Image.LANCZOS)
        image_prompt = np.array(image_prompt)
        image_prompt_mask = np.array(image_prompt_mask)
        
        # 腐蚀再膨胀
        erode_kernel = np.ones((5, 5), np.uint8)
        image_prompt_mask = cv2.erode(image_prompt_mask, erode_kernel)
        dilate_kernel = np.ones((50, 50), np.uint8)
        dilate_mask = cv2.dilate(image_prompt_mask, dilate_kernel)
        dilate_mask = np.uint8(dilate_mask)
        # 得到膨胀后mask对应的ref image的bbox
        coor_image_prompt = np.nonzero(dilate_mask)
        xmin_dilate_mask = coor_image_prompt[0][0]
        xmax_dilate_mask = coor_image_prompt[0][-1]
        coor_image_prompt[1].sort()
        ymin_dilate_mask = coor_image_prompt[1][0]
        ymax_dilate_mask = coor_image_prompt[1][-1]
        
        image_prompt = image_prompt[xmin_dilate_mask:xmax_dilate_mask, ymin_dilate_mask:ymax_dilate_mask, :]
        image_prompt_mask = image_prompt_mask[xmin_dilate_mask:xmax_dilate_mask, ymin_dilate_mask:ymax_dilate_mask]
        image_prompt_np = np.uint8(image_prompt)
        image_prompt_mask_np = np.uint8(image_prompt_mask)
        image_prompt = Image.fromarray(image_prompt_np)
        image_prompt_mask = Image.fromarray(image_prompt_mask_np)

        ip_image_prompt = image_prompt
        # 修改ref image大小以匹配background image mask大小
        blank_image_prompt_mask = Image.new('L', (W, H), 'black')
        image_prompt_w = ymax_dilate_mask - ymin_dilate_mask
        image_prompt_h = xmax_dilate_mask - xmin_dilate_mask

        ref_ratio = image_prompt_w / image_prompt_h
        if mask_w / mask_h > ref_ratio:
            new_h = mask_h
            new_w = new_h * ref_ratio
        else:
            new_w = mask_w
            new_h = new_w / ref_ratio
        image_prompt_h = int(new_h)
        image_prompt_w = int(new_w)
        image_prompt = image_prompt.resize((image_prompt_w, image_prompt_h), Image.LANCZOS)
        image_prompt_mask = image_prompt_mask.resize((image_prompt_w, image_prompt_h), Image.LANCZOS)
        
        set_x = int((mask_h - image_prompt_h) / 2) + xmin
        set_y = int((mask_w - image_prompt_w) / 2) + ymin

        blank_image_prompt = Image.new('RGB', (W, H), 'black')
        blank_image_prompt.paste(image_prompt, (set_y, set_x))
        image_prompt = blank_image_prompt
        blank_image_prompt_mask.paste(image_prompt_mask, (set_y, set_x))
        image_prompt_mask = blank_image_prompt_mask

        # background image mask: 参考物体形状mask
        dilate_mask = dilate_mask[xmin_dilate_mask:xmax_dilate_mask, ymin_dilate_mask:ymax_dilate_mask]
        dilate_mask = Image.fromarray(dilate_mask)
        dilate_mask = dilate_mask.resize((image_prompt_w, image_prompt_h), Image.LANCZOS)
        blank_image_prompt_mask = Image.new('L', (W, H), 'black')
        blank_image_prompt_mask.paste(dilate_mask, (set_y, set_x))
        dilate_mask = blank_image_prompt_mask.convert("RGB")
        mask_image = dilate_mask

        # save blend image
        init_image_np = np.array(init_image)
        blend_image_prompt_np = np.array(image_prompt)
        blend_image_prompt_mask_np = np.array(image_prompt_mask)
        blend_image_prompt_mask_np = np.repeat(blend_image_prompt_mask_np[:, :, np.newaxis], 3, axis=2)
        init_image_blend = init_image_np * (1 - blend_image_prompt_mask_np / 255) + blend_image_prompt_np * (blend_image_prompt_mask_np / 255)
        init_image_blend = init_image_blend.astype(np.uint8)
        init_image_blend_PIL = Image.fromarray(init_image_blend)
        image_prompt = init_image_blend_PIL

        if args.load_controlnet:
            cv_input_image = pil_to_cv2(init_image_blend_PIL)
            detected_map = cv2.Canny(cv_input_image, 100, 200)
            canny_map = Image.fromarray(detected_map)

        sample_ref_match={0 : 0}            
        prompts = [prompt]
        # inversion
        image_gt = [init_image, image_prompt]
        image_gt = [np.array(image) for image in image_gt]
        image_gt = np.stack(image_gt)
        inv_prompts = len(sample_ref_match) * ['']

        end_timestep = ddim_steps - args.threshold_value
        inv_images, x_t, x_stars, ddim_times, prompt_embeds, pooled_prompt_embeds = inversion.invert(
            image_gt, image_prompt_mask, inv_prompts, 
            inv_batch_size=len(inv_prompts), 
            end_timestep=end_timestep)
        print("success inversion")
                
        content_embeds, uncond_content_embeds = get_image_embeds(
                pil_image=ip_image_prompt, 
                device=device, 
                clip_image_processor=clip_image_processor, 
                image_encoder=image_encoder, 
                image_proj_model=image_proj_model,
            )
        style_embeds, uncond_style_embeds = get_image_embeds(
                pil_image=init_image, 
                device=device, 
                clip_image_processor=clip_image_processor, 
                image_encoder=image_encoder, 
                image_proj_model=image_proj_model,
            )
        content_embeds = torch.cat((content_embeds, uncond_content_embeds), dim=1)
        style_embeds = torch.cat((style_embeds, uncond_style_embeds), dim=1)   
        mlp_outputs = mlp_model(content_embeds.to(torch.float32), style_embeds.to(torch.float32))

        label_embeds = content_embeds + style_embeds - mlp_outputs
        image_prompt_embeds = label_embeds[:, :4, :]
        uncond_image_prompt_embeds =label_embeds[:, 4:, :]
        image_prompt_embeds = image_prompt_embeds.to(torch_dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(torch_dtype)
        images = pipe(
                prompt=prompts,
                height=H,
                width=W,
                guidance_scale=7.5, 
                latents=x_t,
                x_stars=x_stars,
                mask=mask_image,
                num_inference_steps=ddim_steps,
                sample_ref_match=sample_ref_match,
                negative_prompt_embeds=prompt_embeds, 
                negative_pooled_prompt_embeds=pooled_prompt_embeds,
                image_prompt_mask=image_prompt_mask,
                text_encoder_inv=text_encoder_inv,
                text_encoder_2_inv=text_encoder_2_inv,
                controlnet_image=canny_map if args.load_controlnet else None,
                controlnet_conditioning_scale=0.2 if args.load_controlnet else None, 
                load_controlnet=args.load_controlnet,
                ip_amend_image_prompt_embeds=image_prompt_embeds,
                ip_amend_uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                threshold_value=args.threshold_value,
                use_adain_timestep_begin=args.use_adain_timestep_begin,
                use_adain_timestep_end=args.use_adain_timestep_end,
                threshold_end=args.threshold_end,
                backgroud_maintain=args.backgroud_maintain
                )                

        images[0] = images[0].resize((orig_w, orig_h), Image.LANCZOS)
        image_id = str(item['id'])
        images[0].save(os.path.join(save_path, f'{image_id}.png'))