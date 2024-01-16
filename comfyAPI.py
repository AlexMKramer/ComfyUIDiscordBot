import uuid
import json
import urllib.request
import urllib.parse


server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break # Execution is done
        else:
            continue # previews are binary data
    print("Execution done, getting images")

    history = get_history(prompt_id)[prompt_id]
    print(history)
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output
    print("Got images")
    return output_images


def get_gifs(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break # Execution is done
        else:
            continue # previews are binary data
    print("Execution done, getting images")

    history = get_history(prompt_id)[prompt_id]
    print(history)
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'gifs' in node_output:
                images_output = []
                for gif in node_output['gifs']:
                    image_data = get_image(gif['filename'], gif['subfolder'], gif['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output
    print("Got images")
    return output_images

prompt_text = """
{
    "4": {
        "inputs": {
            "ckpt_name": "sd_xl_refiner_1.0_0.9vae.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 4
        },
        "class_type": "EmptyLatentImage"
    },
    "8": {
        "inputs": {
            "samples": [
                "23",
                0
            ],
            "vae": [
                "4",
                2
            ]
        },
        "class_type": "VAEDecode"
    },
    "10": {
        "inputs": {
            "ckpt_name": "copaxTimelessxlSDXL1_v42.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "22": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 924819647139783,
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "start_at_step": 0,
            "end_at_step": 13,
            "return_with_leftover_noise": "enable",
            "model": [
                "153",
                0
            ],
            "positive": [
                "75",
                0
            ],
            "negative": [
                "82",
                0
            ],
            "latent_image": [
                "5",
                0
            ]
        },
        "class_type": "KSamplerAdvanced"
    },
    "23": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 924819647139783,
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "start_at_step": 13,
            "end_at_step": 20,
            "return_with_leftover_noise": "disable",
            "model": [
                "4",
                0
            ],
            "positive": [
                "120",
                0
            ],
            "negative": [
                "81",
                0
            ],
            "latent_image": [
                "22",
                0
            ]
        },
        "class_type": "KSamplerAdvanced"
    },
    "75": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": 1024,
            "target_height": 1024,
            "text_g": [
                "153",
                2
            ],
            "text_l": [
                "153",
                2
            ],
            "clip": [
                "153",
                1
            ]
        },
        "class_type": "CLIPTextEncodeSDXL"
    },
    "81": {
        "inputs": {
            "ascore": 1,
            "width": 1024,
            "height": 1024,
            "text": [
                "146",
                1
            ],
            "clip": [
                "4",
                1
            ]
        },
        "class_type": "CLIPTextEncodeSDXLRefiner"
    },
    "82": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": 1024,
            "target_height": 1024,
            "text_g": [
                "146",
                1
            ],
            "text_l": [
                "146",
                1
            ],
            "clip": [
                "153",
                1
            ]
        },
        "class_type": "CLIPTextEncodeSDXL"
    },
    "120": {
        "inputs": {
            "ascore": 6,
            "width": 1024,
            "height": 1024,
            "text": [
                "146",
                0
            ],
            "clip": [
                "4",
                1
            ]
        },
        "class_type": "CLIPTextEncodeSDXLRefiner"
    },
    "122": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "154",
                0
            ]
        },
        "class_type": "SaveImage"
    },
    "146": {
        "inputs": {
            "text_positive": "A serene and contemplative art style, a lone traveler walking along a peaceful path surrounded by lush green hills and blooming flowers, tranquil, hopeful, spiritual, soft earth tones with touches of serene blues and greens",
            "text_negative": "disfigured, ugly, disfigured, gross, nsfw, writing",
            "style": "base",
            "log_prompt": "No"
        },
        "class_type": "SDXLPromptStyler"
    },
    "153": {
        "inputs": {
            "text": [
                "146",
                0
            ],
            "model": [
                "10",
                0
            ],
            "clip": [
                "10",
                1
            ]
        },
        "class_type": "LoraTagLoader"
    },
    "154": {
        "inputs": {
            "upscale_model": [
                "155",
                0
            ],
            "image": [
                "8",
                0
            ]
        },
        "class_type": "ImageUpscaleWithModel"
    },
    "155": {
        "inputs": {
            "model_name": "RealESRGAN_x2.pth"
        },
        "class_type": "UpscaleModelLoader"
    }
}
"""

img2img_prompt_text = """
{
  "4": {
    "inputs": {
      "ckpt_name": "sd_xl_refiner_1.0_0.9vae.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "8": {
    "inputs": {
      "samples": [
        "23",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "10": {
    "inputs": {
      "ckpt_name": "copaxTimelessxlSDXL1_v42.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "22": {
    "inputs": {
      "add_noise": "enable",
      "noise_seed": 860377617572938,
      "steps": 50,
      "cfg": 7.5,
      "sampler_name": "dpmpp_2m",
      "scheduler": "karras",
      "start_at_step": 20,
      "end_at_step": 42,
      "return_with_leftover_noise": "enable",
      "model": [
        "153",
        0
      ],
      "positive": [
        "75",
        0
      ],
      "negative": [
        "82",
        0
      ],
      "latent_image": [
        "157",
        0
      ]
    },
    "class_type": "KSamplerAdvanced"
  },
  "23": {
    "inputs": {
      "add_noise": "disable",
      "noise_seed": 860377617572938,
      "steps": 50,
      "cfg": 7.5,
      "sampler_name": "dpmpp_2m",
      "scheduler": "karras",
      "start_at_step": 42,
      "end_at_step": 50,
      "return_with_leftover_noise": "disable",
      "model": [
        "4",
        0
      ],
      "positive": [
        "120",
        0
      ],
      "negative": [
        "81",
        0
      ],
      "latent_image": [
        "22",
        0
      ]
    },
    "class_type": "KSamplerAdvanced"
  },
  "75": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "crop_w": 0,
      "crop_h": 0,
      "target_width": 1024,
      "target_height": 1024,
      "text_g": [
        "153",
        2
      ],
      "text_l": [
        "153",
        2
      ],
      "clip": [
        "153",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXL"
  },
  "81": {
    "inputs": {
      "ascore": 1,
      "width": 1024,
      "height": 1024,
      "text": [
        "146",
        1
      ],
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner"
  },
  "82": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "crop_w": 0,
      "crop_h": 0,
      "target_width": 1024,
      "target_height": 1024,
      "text_g": [
        "146",
        1
      ],
      "text_l": [
        "146",
        1
      ],
      "clip": [
        "153",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXL"
  },
  "120": {
    "inputs": {
      "ascore": 6,
      "width": 1024,
      "height": 1024,
      "text": [
        "146",
        0
      ],
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncodeSDXLRefiner"
  },
  "122": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "154",
        0
      ]
    },
    "class_type": "SaveImage"
  },
  "146": {
    "inputs": {
      "text_positive": "lich kind in a forest, dark, cinematic film still, scary",
      "text_negative": "disfigured, ugly, disfigured, gross, nsfw, writing",
      "style": "base",
      "log_prompt": "No"
    },
    "class_type": "SDXLPromptStyler"
  },
  "153": {
    "inputs": {
      "text": [
        "146",
        0
      ],
      "model": [
        "10",
        0
      ],
      "clip": [
        "10",
        1
      ]
    },
    "class_type": "LoraTagLoader"
  },
  "154": {
    "inputs": {
      "upscale_model": [
        "155",
        0
      ],
      "image": [
        "8",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel"
  },
  "155": {
    "inputs": {
      "model_name": "4x-UltraSharp.pth"
    },
    "class_type": "UpscaleModelLoader"
  },
  "156": {
    "inputs": {
      "image": "temp_image.png",
      "choose file to upload": "image"
    },
    "class_type": "LoadImage"
  },
  "157": {
    "inputs": {
      "pixels": [
        "160",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncode"
  },
  "160": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "width": 1024,
      "height": 1024,
      "crop": "disabled",
      "image": [
        "156",
        0
      ]
    },
    "class_type": "ImageScale"
  }
}
"""

upscale_prompt_text = """
{
  "1": {
    "inputs": {
      "model_name": "4x-UltraSharp.pth"
    },
    "class_type": "UpscaleModelLoader"
  },
  "2": {
    "inputs": {
      "upscale_model": [
        "1",
        0
      ],
      "image": [
        "3",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel"
  },
  "3": {
    "inputs": {
      "image": "temp_upscale.png",
      "choose file to upload": "image"
    },
    "class_type": "LoadImage"
  },
  "4": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "2",
        0
      ]
    },
    "class_type": "SaveImage"
  }
}
"""

turbo_prompt_text = """
{
  "5": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 4
    },
    "class_type": "EmptyLatentImage"
  },
  "6": {
    "inputs": {
      "text": "landscape photograph of a forest with mountains",
      "clip": [
        "20",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "text, watermark",
      "clip": [
        "20",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "20",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "13": {
    "inputs": {
      "add_noise": true,
      "noise_seed": 2,
      "cfg": 1,
      "model": [
        "20",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "sampler": [
        "14",
        0
      ],
      "sigmas": [
        "22",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "SamplerCustom"
  },
  "14": {
    "inputs": {
      "sampler_name": "dpmpp_2m_sde"
    },
    "class_type": "KSamplerSelect"
  },
  "20": {
    "inputs": {
      "ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "22": {
    "inputs": {
      "steps": 4,
      "model": [
        "20",
        0
      ]
    },
    "class_type": "SDTurboScheduler"
  },
  "27": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage"
  }
}
"""

txt2vid_prompt_text = """
{
  "2": {
    "inputs": {
      "vae_name": "vae-ft-840000-ema-pruned.ckpt"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "3": {
    "inputs": {
      "text": "Donald Trump eating spaghetti and meatballs, masterpiece, oil painting",
      "clip": [
        "4",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "4": {
    "inputs": {
      "stop_at_clip_layer": -2,
      "clip": [
        "32",
        1
      ]
    },
    "class_type": "CLIPSetLastLayer",
    "_meta": {
      "title": "CLIP Set Last Layer"
    }
  },
  "6": {
    "inputs": {
      "text": "(worst quality, low quality: 1.4)",
      "clip": [
        "4",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "seed": 452371923286995,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "36",
        0
      ],
      "positive": [
        "3",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "latent_image": [
        "9",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "9": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 48
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "10": {
    "inputs": {
      "samples": [
        "7",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "32": {
    "inputs": {
      "ckpt_name": "sd-1.5/sd-1.5_checkpoints/noosphere_v4.2.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "33": {
    "inputs": {
      "context_length": 16,
      "context_stride": 1,
      "context_overlap": 4,
      "context_schedule": "uniform",
      "closed_loop": false
    },
    "class_type": "ADE_AnimateDiffUniformContextOptions",
    "_meta": {
      "title": "Uniform Context Options 🎭🅐🅓"
    }
  },
  "36": {
    "inputs": {
      "model_name": "v3_sd15_mm.ckpt",
      "beta_schedule": "sqrt_linear (AnimateDiff)",
      "motion_scale": 1,
      "apply_v2_models_properly": true,
      "model": [
        "32",
        0
      ],
      "context_options": [
        "33",
        0
      ]
    },
    "class_type": "ADE_AnimateDiffLoaderWithContext",
    "_meta": {
      "title": "AnimateDiff Loader 🎭🅐🅓"
    }
  },
  "37": {
    "inputs": {
      "frame_rate": 8,
      "loop_count": 0,
      "filename_prefix": "aaa_readme",
      "format": "image/gif",
      "pingpong": false,
      "save_output": true,
      "images": [
        "10",
        0
      ]
    },
    "class_type": "VHS_VideoCombine",
    "_meta": {
      "title": "Video Combine 🎥🅥🅗🅢"
    }
  }
}
"""

prompt = json.loads(prompt_text)

img2img_prompt = json.loads(img2img_prompt_text)

upscale_prompt = json.loads(upscale_prompt_text)

turbo_prompt = json.loads(turbo_prompt_text)

txt2vid_prompt = json.loads(txt2vid_prompt_text)
