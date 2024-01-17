import asyncio
import uuid
import json
import urllib.request
import urllib.parse
import time
import requests

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
        print("http://{}/view?{}".format(server_address, url_values))
        return response.read()


def get_gif_url(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url = "http://{}/view".format(server_address)

    try:
        response = requests.get(url, params=data)
        response.raise_for_status()

        print(response.url)
        return response.url

    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        return None


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
    gif_urls = {}
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

    # wait for the gif to be generated and combined
    print('Waiting for gif to be generated and combined')
    time.sleep(10)

    history = get_history(prompt_id)[prompt_id]
    print(history)
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'gifs' in node_output:
                images_output = []
                for gif in node_output['gifs']:
                    gif_urls = get_gif_url(gif['filename'], gif['subfolder'], gif['type'])

    print("Got gif urls")
    return gif_urls


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
      "text": "Donald Trump eating spaghetti and meatballs",
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
      "seed": 888888889,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "27",
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
      "batch_size": 16
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
  "27": {
    "inputs": {
      "model_name": "v3_sd15_mm.ckpt",
      "beta_schedule": "sqrt_linear (AnimateDiff)",
      "motion_scale": 1,
      "apply_v2_models_properly": true,
      "model": [
        "32",
        0
      ]
    },
    "class_type": "ADE_AnimateDiffLoaderWithContext",
    "_meta": {
      "title": "AnimateDiff Loader 🎭🅐🅓"
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
  "35": {
    "inputs": {
      "frame_rate": 8,
      "loop_count": 0,
      "filename_prefix": "ComfyuiGif",
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

high_quality_prompt_text = """
{
  "1": {
    "inputs": {
      "ckpt_name": "DreamShaperXL_turbo.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "3": {
    "inputs": {
      "text": [
        "102",
        2
      ],
      "clip": [
        "102",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "4": {
    "inputs": {
      "text": [
        "100",
        2
      ],
      "clip": [
        "100",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "5": {
    "inputs": {
      "seed": 351988000611512,
      "steps": 8,
      "cfg": 2,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "karras",
      "denoise": 1,
      "model": [
        "102",
        0
      ],
      "positive": [
        "3",
        0
      ],
      "negative": [
        "4",
        0
      ],
      "latent_image": [
        "97",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "7": {
    "inputs": {
      "samples": [
        "5",
        0
      ],
      "vae": [
        "1",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "11": {
    "inputs": {
      "max_faces": 10,
      "min_confidence": 0.5,
      "resolution": 512,
      "image": [
        "73",
        0
      ]
    },
    "class_type": "MediaPipe-FaceMeshPreprocessor",
    "_meta": {
      "title": "MediaPipe Face Mesh"
    }
  },
  "13": {
    "inputs": {
      "crop_factor": 3,
      "bbox_fill": false,
      "crop_min_size": 50,
      "drop_size": 1,
      "dilation": 0,
      "face": false,
      "mouth": false,
      "left_eyebrow": false,
      "left_eye": true,
      "left_pupil": false,
      "right_eyebrow": false,
      "right_eye": true,
      "right_pupil": false,
      "image": [
        "11",
        0
      ]
    },
    "class_type": "MediaPipeFaceMeshToSEGS",
    "_meta": {
      "title": "MediaPipe FaceMesh to SEGS"
    }
  },
  "15": {
    "inputs": {
      "segs": [
        "13",
        0
      ]
    },
    "class_type": "SegsToCombinedMask",
    "_meta": {
      "title": "SEGS to MASK (combined)"
    }
  },
  "27": {
    "inputs": {
      "guide_size": 512,
      "guide_size_for": true,
      "max_size": 1536,
      "seed": 351988000611512,
      "steps": 5,
      "cfg": 2,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "karras",
      "denoise": 0.4,
      "feather": 5,
      "noise_mask": false,
      "force_inpaint": true,
      "wildcard": "",
      "cycle": 1,
      "inpaint_model": false,
      "noise_mask_feather": 10,
      "image": [
        "73",
        0
      ],
      "segs": [
        "41",
        0
      ],
      "model": [
        "102",
        0
      ],
      "clip": [
        "100",
        1
      ],
      "vae": [
        "1",
        2
      ],
      "positive": [
        "3",
        0
      ],
      "negative": [
        "4",
        0
      ]
    },
    "class_type": "DetailerForEachDebug",
    "_meta": {
      "title": "DetailerDebug (SEGS)"
    }
  },
  "41": {
    "inputs": {
      "combined": true,
      "crop_factor": 3,
      "bbox_fill": false,
      "drop_size": 10,
      "contour_fill": false,
      "mask": [
        "15",
        0
      ]
    },
    "class_type": "MaskToSEGS",
    "_meta": {
      "title": "MASK to SEGS"
    }
  },
  "54": {
    "inputs": {
      "model_name": "bbox/hand_yolov8s.pt"
    },
    "class_type": "UltralyticsDetectorProvider",
    "_meta": {
      "title": "UltralyticsDetectorProvider"
    }
  },
  "61": {
    "inputs": {
      "threshold": 0.5,
      "dilation": 30,
      "crop_factor": 3,
      "drop_size": 10,
      "labels": "all",
      "bbox_detector": [
        "54",
        0
      ],
      "image": [
        "7",
        0
      ]
    },
    "class_type": "BboxDetectorSEGS",
    "_meta": {
      "title": "BBOX Detector (SEGS)"
    }
  },
  "67": {
    "inputs": {
      "guide_size": 512,
      "guide_size_for": true,
      "max_size": 1536,
      "seed": 351988000611512,
      "steps": 5,
      "cfg": 2,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "karras",
      "denoise": 0.4,
      "feather": 20,
      "noise_mask": true,
      "force_inpaint": true,
      "wildcard": "",
      "cycle": 1,
      "inpaint_model": false,
      "noise_mask_feather": 10,
      "image": [
        "27",
        0
      ],
      "segs": [
        "61",
        0
      ],
      "model": [
        "102",
        0
      ],
      "clip": [
        "1",
        1
      ],
      "vae": [
        "1",
        2
      ],
      "positive": [
        "3",
        0
      ],
      "negative": [
        "4",
        0
      ]
    },
    "class_type": "DetailerForEachDebug",
    "_meta": {
      "title": "DetailerDebug (SEGS)"
    }
  },
  "71": {
    "inputs": {
      "filename_prefix": "Comfyui",
      "images": [
        "67",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "72": {
    "inputs": {
      "seed": 351988000611512,
      "steps": 5,
      "cfg": 2,
      "sampler_name": "dpmpp_sde_gpu",
      "scheduler": "karras",
      "denoise": 0.45,
      "model": [
        "102",
        0
      ],
      "positive": [
        "3",
        0
      ],
      "negative": [
        "4",
        0
      ],
      "latent_image": [
        "75",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "73": {
    "inputs": {
      "samples": [
        "72",
        0
      ],
      "vae": [
        "1",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "75": {
    "inputs": {
      "pixels": [
        "79",
        0
      ],
      "vae": [
        "1",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "76": {
    "inputs": {
      "model_name": "4x-UltraSharp.pth"
    },
    "class_type": "UpscaleModelLoader",
    "_meta": {
      "title": "Load Upscale Model"
    }
  },
  "77": {
    "inputs": {
      "upscale_model": [
        "76",
        0
      ],
      "image": [
        "7",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel",
    "_meta": {
      "title": "Upscale Image (using Model)"
    }
  },
  "79": {
    "inputs": {
      "upscale_method": "lanczos",
      "scale_by": 0.25,
      "image": [
        "77",
        0
      ]
    },
    "class_type": "ImageScaleBy",
    "_meta": {
      "title": "Upscale Image By"
    }
  },
  "96": {
    "inputs": {
      "images": [
        "67",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image (Detailer)"
    }
  },
  "97": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "100": {
    "inputs": {
      "text": [
        "101",
        1
      ],
      "model": [
        "1",
        0
      ],
      "clip": [
        "1",
        1
      ]
    },
    "class_type": "LoraTagLoader",
    "_meta": {
      "title": "Load LoRA Tag Negative"
    }
  },
  "101": {
    "inputs": {
      "text_positive": "high-resolution digital painting of a mystical forest transforming at twilight. The style is surreal and ethereal, reminiscent of the Romanticism era, emphasizing the sublime beauty of nature. The scene features glowing flowers, ethereal lights, and subtly visible mythical creatures. The color palette consists of twilight hues â purples, blues, and soft pinks, with glowing accents. Lighting is soft and ethereal, highlighting the magical transformation of the forest under the moonlight.",
      "text_negative": "",
      "style": "base",
      "log_prompt": true
    },
    "class_type": "SDXLPromptStyler",
    "_meta": {
      "title": "SDXL Prompt Styler"
    }
  },
  "102": {
    "inputs": {
      "text": [
        "101",
        0
      ],
      "model": [
        "1",
        0
      ],
      "clip": [
        "1",
        1
      ]
    },
    "class_type": "LoraTagLoader",
    "_meta": {
      "title": "Load LoRA Tag Positive"
    }
  }
}
"""

prompt = json.loads(prompt_text)

img2img_prompt = json.loads(img2img_prompt_text)

upscale_prompt = json.loads(upscale_prompt_text)

turbo_prompt = json.loads(turbo_prompt_text)

txt2vid_prompt = json.loads(txt2vid_prompt_text)

high_quality_prompt = json.loads(high_quality_prompt_text)
