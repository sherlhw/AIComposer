import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Union, Tuple, List
from tqdm import tqdm
import os
import cv2
from diffusers import DDIMInverseScheduler, DPMSolverMultistepInverseScheduler
class Inversion:

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        # self.scheduler.final_alpha_cumprod
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else 0.9991
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def get_noise_pred_single(self, latents, t, context, cond=True, both=False):
        added_cond_id = 1 if cond else 0
        do_classifier_free_guidance = False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        if both is False:
            added_cond_kwargs = {
                "text_embeds": self.add_text_embeds[added_cond_id].unsqueeze(0).repeat(self.inv_batch_size, 1), 
                "time_ids": self.add_time_ids[added_cond_id].unsqueeze(0).repeat(self.inv_batch_size, 1)
                }
        else:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=context,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        self.model.vae.to(dtype=torch.float32)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            else:
                if image.ndim==3:
                    image=np.expand_dims(image,0)
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(self.device)
                latents=[]
                for i, _ in enumerate(image):
                    latent=self.model.vae.encode(image[i:i+1])['latent_dist'].mean
                    latents.append(latent)
                latents = torch.stack(latents).squeeze(1)
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(
        self,
        prompt:  Union[str, List[str]],
        prompt_2:  Optional[Union[str, List[str]]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):  
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        original_size = original_size or (1024, 1024)
        target_size = target_size or (1024, 1024)
        # 3. Encode input prompt
        do_classifier_free_guidance=True
        if self.text_encoder_inv is None:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.model.encode_prompt(
                prompt,
                prompt_2,
                self.model.device,
                1,
                do_classifier_free_guidance,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
            )
        else:
            # 使用7788作为prompt
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.model.encode_prompt_null_prompt(
                prompt,
                prompt_2,
                self.model.device,
                1,
                do_classifier_free_guidance,
                negative_prompt=None,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
                text_encoder_inv=self.text_encoder_inv,
                text_encoder_2_inv=self.text_encoder_2_inv,
            )
        prompt_embeds=prompt_embeds[:self.inv_batch_size]
        negative_prompt_embeds=negative_prompt_embeds[:self.inv_batch_size]
        pooled_prompt_embeds=pooled_prompt_embeds[:self.inv_batch_size]
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[:self.inv_batch_size]
        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
    
        if self.model.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim
        add_time_ids = self.model._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        self.add_text_embeds = add_text_embeds.to(self.device)
        self.add_time_ids = add_time_ids.to(self.device).repeat(self.inv_batch_size * 1, 1)

        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        self.prompt = prompt
        self.context = prompt_embeds

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        all_time = [0]
        latent = latent.clone().detach()
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(self.generator, self.eta)
        if isinstance(self.inverse_scheduler, DDIMInverseScheduler):
            extra_step_kwargs.pop("generator")
        if self.end_timestep == None:
            self.end_timestep = self.num_ddim_steps
        for i in range(self.end_timestep):
            use_inv_sc = False 
            if use_inv_sc:
                t = self.inverse_scheduler.timesteps[i]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, cond=True)
                latent = self.inverse_scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            else:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, cond=True)
                latent = self.next_step(noise_pred, t, latent)
            all_time.append(t.item())
            all_latent.append(latent)

        return all_latent, all_time

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, device='cuda', epsilon=1):
        latent = self.image2latent(image) 
        latent = latent.to(device) 

        orig_mask = np.array(self.image_prompt_mask)
        # use adain
        self.image_prompt_mask = self.image_prompt_mask.convert('L')
        self.image_prompt_mask = self.image_prompt_mask.resize((latent.shape[-1], latent.shape[-2]))
        image_prompt_mask = np.array(self.image_prompt_mask) / 255
        image_prompt_mask = torch.tensor(image_prompt_mask, device=device, dtype=latent.dtype)
        image_prompt_mask = image_prompt_mask.bool()
        epsilon = epsilon
        latent_bg = latent[0]
        latent_fg = latent[1]
        latent_fg[:, image_prompt_mask] = self.model.adaptive_latent_normalization(latent_fg, latent_bg, image_prompt_mask, epsilon)
        latent[1] = latent_fg
        images = self.latent2image(latent)
        orig_mask = np.repeat(orig_mask[:, :, np.newaxis], 3, axis=2)
        image[0] = images[0] * (1 - orig_mask / 255) + images[1] * (orig_mask / 255)
        latent = self.image2latent(image[:1])
        image = self.latent2image(latent)
            
        ddim_latents, ddim_time = self.ddim_loop(latent.to(self.model.unet.dtype)) 
        
        return image, ddim_latents, ddim_time

    from typing import Union, List, Dict
    import numpy as np

    def invert(self, image_gt, image_prompt_mask=None, prompt=None, 
            verbose=True, 
            inv_output_pos=None,
            inv_batch_size=1,
            adain_epsilon=1,
            end_timestep=None
        ):
        self.end_timestep = end_timestep
        self.inv_batch_size = inv_batch_size
        self.init_prompt(prompt)
        out_put_pos = 0 if inv_output_pos is None else inv_output_pos
        self.out_put_pos = out_put_pos
        self.image_prompt_mask = image_prompt_mask
        if verbose:
            print("inversion...")
        image_rec, ddim_latents, ddim_times = self.ddim_inversion(image_gt, epsilon=adain_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], ddim_latents, ddim_times, self.prompt_embeds[self.prompt_embeds.shape[0]//2:], self.pooled_prompt_embeds

    def __init__(self, model, num_ddim_steps, generator=None, scheduler_type="DDIM", text_encoder_inv=None, text_encoder_2_inv=None):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.num_ddim_steps=num_ddim_steps
        if scheduler_type == "DDIM":
            self.inverse_scheduler=DDIMInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        elif scheduler_type=="DPMSolver":
            self.inverse_scheduler=DPMSolverMultistepInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.model.vae.to(dtype=torch.float32)
        self.prompt = None
        self.context = None
        self.device=self.model.unet.device
        self.generator=generator
        self.eta=0.0
        self.text_encoder_inv = text_encoder_inv
        self.text_encoder_2_inv = text_encoder_2_inv

def load_1024_mask(image_path, left=0, right=0, top=0, bottom=0,target_H=128,target_W=128):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, np.newaxis]
    else:
        image = image_path
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image=image.squeeze()
    image = np.array(Image.fromarray(image).resize((target_H, target_W)))
    return image

def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).resize((1024, 1024)))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image