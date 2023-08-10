# ComfyUIDiscordBot
## A Discord bot for the ComfyUI Discord server.

This is my very basic attempt at making a discord bot for ComfyUI. It's written in Python and uses the py-cord library.
I am not a programmer by any means, so this is probably not the best code you'll ever see. I'm just doing this for fun.
Any suggestions or help is welcome.

### Requirements
- ComfyUI running on a machine.  This is setup for running it on the same machine, but can be configured to use another machine by changing the "server_address" variable in comfyAPI.py.
- SDXL Base and Refiner models.  These can be found here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors and here: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors
- ComfyUI custom node: SDXL Prompt Styler found here: https://github.com/twri/sdxl_prompt_styler
- Python and pip.

### Models
Currently, the bot uses the SDXL Base and the SDXL Refiner.  I am using the 1.0 version with the 0.9 VAE baked in.

These are named sd_xl_base_1.0_0.9vae.safetensors and sd_xl_refiner_1.0_0.9vae.safetensors respectively.

If you want to use different models, change the two model names below.  I recommend using the combination of Base and Refiner, as used by this bot.

### Commands
- [x] **Draw** - `/draw` `new prompt` `new style` `new height width`

    Draw will generate 4 images based on the prompt you give it, with an optional style and size.  If no style is given, it will use the default style.  If no size is given, it will use the default size.
- [x] **Crazy** - `/crazy`

    Crazy will generate 4 images using a random subject, activity, and location with a random style.
- [ ] **Upscale**

  


## Installation

### Clone the repository.

    git clone https://github.com/comfyanonymous/ComfyUI.git

I recommend using a virtual environment.  This can be done by running the following commands in the root directory of the bot;

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt

### Update sdxl_styles.json
Replace the sdxl_styles.json in ComfyUI/custom_nodes/sdxl_prompt_styler/ with the one found in this repository.

### Update main.py
You will need to fill in several variables in main.py;
- `TOKEN` - Your bot's token
- `folder_path` - The path where your ComfyUI checkpoints folder is located. Ex; /home/USER/ComfyUI/checkpoints
- `base_mode` - The name of the base model. Ex; sd_xl_base_1.0_0.9vae.safetensors
- `refiner_model` - The name of the refiner model. Ex; sd_xl_refiner_1.0_0.9vae.safetensors


# Thanks
/Rotyxium/CtoD for the ground work on the bot.