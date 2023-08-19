import uuid
import json
import urllib.request
import urllib.parse


server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
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

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

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
            "noise_seed": 614487432439792,
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
            "noise_seed": 614487432439792,
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
            "width": 4096,
            "height": 4096,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": 4096,
            "target_height": 4096,
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
            "width": 4096,
            "height": 4096,
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
            "width": 4096,
            "height": 4096,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": 4096,
            "target_height": 4096,
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
            "width": 4096,
            "height": 4096,
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
    "146": {
        "inputs": {
            "text_positive": "A dreamlike and ethereal art style, a couple standing on a cliff overlooking a vast, otherworldly landscape; they are holding each other close, finding solace and strength in each other's presence; the setting sun casts a warm, golden light on their faces, a color palette of soft pastels, with hints of warm oranges and pinks, creating a sense of tranquility and hope amidst the unknown",
            "text_negative": "",
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
    "160": {
        "inputs": {
            "model_name": "RealESRGAN_x2.pth"
        },
        "class_type": "UpscaleModelLoader"
    },
    "163": {
        "inputs": {
            "upscale_model": [
                "160",
                0
            ],
            "image": [
                "169",
                0
            ]
        },
        "class_type": "ImageUpscaleWithModel"
    },
    "165": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "163",
                0
            ]
        },
        "class_type": "SaveImage"
    },
    "169": {
        "inputs": {
            "guide_size": 256,
            "guide_size_for": true,
            "max_size": 768,
            "seed": 784051626223992,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 0.5,
            "feather": 5,
            "noise_mask": true,
            "force_inpaint": false,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "image": [
                "173",
                0
            ],
            "detailer_pipe": [
                "173",
                4
            ]
        },
        "class_type": "FaceDetailerPipe"
    },
    "173": {
        "inputs": {
            "guide_size": 256,
            "guide_size_for": true,
            "max_size": 768,
            "seed": 328905338460486,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 0.5,
            "feather": 5,
            "noise_mask": true,
            "force_inpaint": true,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "wildcard": "",
            "image": [
                "188",
                0
            ],
            "model": [
                "4",
                0
            ],
            "clip": [
                "4",
                1
            ],
            "vae": [
                "4",
                2
            ],
            "positive": [
                "120",
                0
            ],
            "negative": [
                "81",
                0
            ],
            "bbox_detector": [
                "182",
                0
            ],
            "sam_model_opt": [
                "183",
                0
            ],
            "segm_detector_opt": [
                "182",
                1
            ]
        },
        "class_type": "FaceDetailer"
    },
    "182": {
        "inputs": {
            "model_name": "bbox/face_yolov8m.pt"
        },
        "class_type": "UltralyticsDetectorProvider"
    },
    "183": {
        "inputs": {
            "model_name": "sam_vit_b_01ec64.pth",
            "device_mode": "AUTO"
        },
        "class_type": "SAMLoader"
    },
    "188": {
        "inputs": {
            "image": [
                "8",
                0
            ]
        },
        "class_type": "ImpactImageBatchToImageList"
    }
}
"""


prompt = json.loads(prompt_text)

