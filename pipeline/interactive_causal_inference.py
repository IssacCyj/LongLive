# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        #self._recache_generator = torch.compile(self.generator, mode="max-autotune")

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        #num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        #recache_start_frame = current_start_frame - num_recache_frames
        
        #frames_to_recache = output[:, recache_start_frame:current_start_frame]
        
        recache_stride = 2
        if self.local_attn_size == -1:
            num_recache_frames = current_start_frame
            recache_start_frame = 0
            frames_to_recache = output[:, recache_start_frame:current_start_frame]
        else:
            indices = []
            for i in range(self.local_attn_size):
                frame_idx = current_start_frame - 1 - i * recache_stride
                if frame_idx < 0:
                    break
                indices.insert(0, frame_idx)
            
            if not indices:
                return

            frames_to_recache = output[:, indices]
            recache_start_frame = indices[0]
            num_recache_frames = len(indices)

        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            #self._recache_generator(
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        
        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            denoising_step_times = [[] for _ in range(len(self.denoising_step_list))]
            kv_update_times = []
            recache_times = []
            denoising_step_start = torch.cuda.Event(enable_timing=True)
            denoising_step_end = torch.cuda.Event(enable_timing=True)
            kv_update_start = torch.cuda.Event(enable_timing=True)
            kv_update_end = torch.cuda.Event(enable_timing=True)
            recache_start = torch.cuda.Event(enable_timing=True)
            recache_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                if profile:
                    recache_start.record()
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                if profile:
                    recache_end.record()
                    torch.cuda.synchronize()
                    recache_times.append(recache_start.elapsed_time(recache_end))
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                if profile:
                    denoising_step_start.record()
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                if profile:
                    denoising_step_end.record()
                    torch.cuda.synchronize()
                    denoising_step_times[index].append(denoising_step_start.elapsed_time(denoising_step_end))

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            if profile:
                kv_update_start.record()

            # rerun with clean context to update cache
            #context_timestep = torch.ones_like(timestep) * self.args.context_noise
            #self.generator(
            #    noisy_image_or_video=denoised_pred,
            #    conditional_dict=cond_in_use,
            #    timestep=context_timestep,
            #    kv_cache=self.kv_cache1,
            #    crossattn_cache=self.crossattn_cache,
            #    current_start=current_start_frame * self.frame_seq_length,
            #)
            if profile:
                kv_update_end.record()
                torch.cuda.synchronize()
                kv_update_times.append(kv_update_start.elapsed_time(kv_update_end))

                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))

            # Update frame pointer
            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            if block_times:
                avg_block_time = sum(block_times) / len(block_times)
                print(f"    - Average block generation time ({self.num_frame_per_block} frames): {avg_block_time:.2f} ms")
                print(f"    - Average per-frame latency: {avg_block_time / self.num_frame_per_block:.2f} ms")
                print(f"    - Block generation times (ms): {block_times}")

            for i in range(len(denoising_step_times)):
                if denoising_step_times[i]:
                    avg_step_time = sum(denoising_step_times[i]) / len(denoising_step_times[i])
                    print(f"    - Denoising step {i} average time: {avg_step_time:.2f} ms")
                    print(f"    - Denoising step {i} times (ms): {denoising_step_times[i]}")
            if kv_update_times:
                avg_kv_update_time = sum(kv_update_times) / len(kv_update_times)
                print(f"    - KV update step average time: {avg_kv_update_time:.2f} ms")
                print(f"    - KV update step times (ms): {kv_update_times}")
            if recache_times:
                print(f"  - Prompt-switch recache time(s): {recache_times} ms")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

            # Plotting
            try:
                output_folder = '/home/yujiachen/LongLive/plots'
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                if block_times:
                    plt.figure()
                    plt.plot(block_times)
                    plt.xlabel("Block index")
                    plt.ylabel("Latency (ms)")
                    plt.title("Interactive Block Generation Latency")
                    plt.savefig(os.path.join(output_folder, "interactive_block_latency.png"))
                    plt.close()

                if kv_update_times:
                    plt.figure()
                    plt.plot(kv_update_times)
                    plt.xlabel("Block index")
                    plt.ylabel("Latency (ms)")
                    plt.title("Interactive KV Update Latency")
                    plt.savefig(os.path.join(output_folder, "interactive_kv_update_latency.png"))
                    plt.close()

                plt.figure()
                for i in range(len(denoising_step_times)):
                    if denoising_step_times[i]:
                        plt.plot(denoising_step_times[i], label=f"Step {i}")
                plt.xlabel("Block index")
                plt.ylabel("Latency (ms)")
                plt.title("Interactive Denoising Step Latency")
                plt.legend()
                plt.savefig(os.path.join(output_folder, "interactive_denoising_step_latency.png"))
                plt.close()
                print(f"Saved latency plots to {output_folder}")
            except Exception as e:
                print(f"Error during plotting: {e}")
        return video 
