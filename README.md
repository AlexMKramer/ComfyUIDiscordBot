# ComfyUIDiscordBot
## A Discord bot for the ComfyUI Discord server.

This is my very basic attempt at making a discord bot for ComfyUI. It's written in Python and uses the py-cord library.
I am not a programmer by any means, so this is probably not the best code you'll ever see. I'm just doing this for fun.
Any suggestions or help is welcome.

### Requirements
- ComfyUI running on a machine.  This is setup for running it on the same machine, but can be configured to use another machine by changing the "server_address" variable in comfyAPI.py.
- SDXL Base and Refiner models.  These can be found here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors and here: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors
- Python and pip.
- OpenAI API key.  You can get one here: https://beta.openai.com/

#### ComfyUI Custom nodes
- SDXL Prompt Styler: https://github.com/twri/sdxl_prompt_styler
- ComfyUI Impact Pack: https://github.com/ltdrdata/ComfyUI-Impact-Pack
- ComfyUI Lora Tag Loader: https://github.com/badjeff/comfyui_lora_tag_loader/

### Models
Currently, the bot uses the SDXL Base and the SDXL Refiner.  I am using the 1.0 version with the 0.9 VAE baked in.

These are named sd_xl_base_1.0_0.9vae.safetensors and sd_xl_refiner_1.0_0.9vae.safetensors respectively.

If you want to use different models, change the two model names below.  I recommend using the combination of Base and Refiner, as used by this bot.

### Commands
- [x] **Draw** - `/draw` `new prompt` `new style` `new height width`

    Draw will generate 4 images based on the prompt you give it, with an optional style and size.  If no style is given, it will use the default style.  If no size is given, it will use the default size.
- [x] **Crazy** - `/crazy`

    Crazy will generate 4 images using a random subject, activity, and location with a random style.
- [x] **Upscale**

    Upscale will take an image and upscale it using the default style and size in either 2x or 4x.
- [x] **Lora Support**

    Under Draw, Lora will generate an image using the Lora Tag Loader node.  This will take a list of your current loras downloaded in Comfy and add them to the begining of the prompt.
- [x] **Music**

    Music will generate an image by the lyrics using song_name and artist_name.  This will use the default style and size.
- [x] **Interpret**

    Interpret will generate an image using a Song and Artist name to gather the lyrics, that then get sent to ChatGPT to form a prompt.


## Installation

### Clone the repository.

    git clone https://github.com/comfyanonymous/ComfyUI.git

I recommend using a virtual environment.  This can be done by running the following commands in the root directory of the bot;

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
Edit the .env file and fill in the variables.  You will need to create a discord bot and get the token from the discord developer portal. Then fill in the Folder path with the path leading to your ComfyUI/models folder.

- `TOKEN` - Your bot's token
- `folder_path` - The path where your ComfyUI models folder is located. Ex; /home/USER/ComfyUI/models
- `genius_token` - Your genius token.  This is used for the music command.  You can get one here: https://genius.com/api-clients/new
- `OPENAI_API_KEY` - Your OpenAI API key.  You can get one here: https://beta.openai.com/

### Update sdxl_styles.json
Copy the sdxl_styles.json from this repo into the ComfyUI/custom_nodes/sdxl_prompt_styler folder.  This will add the styles to the SDXL Prompt Styler node.

### Update main.py
You will need to fill in several variables in main.py;
- `base_mode` - The name of the base model. Ex; sd_xl_base_1.0_0.9vae.safetensors
- `refiner_model` - The name of the refiner model. Ex; sd_xl_refiner_1.0_0.9vae.safetensors


# Thanks
/Rotyxium/CtoD for the groundwork on the bot.