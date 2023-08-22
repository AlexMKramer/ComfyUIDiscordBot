# ComfyUIDiscordBot
## A Discord bot for the ComfyUI Discord server.

This is my very basic attempt at making a discord bot for ComfyUI. It's written in Python and uses the py-cord library.
I am not a programmer by any means, so this is probably not the best code you'll ever see. I'm just doing this for fun.
Any suggestions or help is welcome.

### Requirements
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running on a machine.  This is set up for running it on the same machine, but can be configured to use another machine by changing the "server_address" variable in comfyAPI.py.
- SDXL [Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) and [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) models.
- Python and pip.
- OpenAI API key.  You can get one [here.](https://beta.openai.com/)

#### ComfyUI Custom nodes
- [SDXL Prompt Styler](https://github.com/twri/sdxl_prompt_styler)
- [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI Lora Tag Loader](https://github.com/badjeff/comfyui_lora_tag_loader/)

### Models, Loras, and Styles
### Models:
Currently, the bot uses the SDXL Base and the SDXL Refiner.  I am using the 1.0 version with the 0.9 VAE baked in.

These are named sd_xl_base_1.0_0.9vae.safetensors and sd_xl_refiner_1.0_0.9vae.safetensors respectively.

If you want to use different models, change the two model names below.  I recommend using the combination of Base and Refiner, as used by this bot.

The bot will search through the models folder in ComfyUI and list each in the `model_name` option of commands.
### Loras:
The bot will search through the Loras folder in ComfyUI and list each in the `new_lora` option of commands.

Loras are added to the prompt at the beginning.  You should still use the proper tags in your prompt to ensure it works properly.
### Styles:
Styles are loaded from the sdxl_styles.json file.  The one included in this repo is not the same in the custom_node "sdxl_prompt_styler".  You will need to copy the one from this repo into the `ComfyUI/custom_node/sdxl_prompt_styler` folder to use the styles in the bot properly.

## Commands
### Draw
Draw will generate 4 images based on the prompt you give it, with an optional style and size.  If no style is given, it will use the default style.  If no size is given, it will use the default size.
#### Usage
    /draw 
#### Required Arguments
    new_prompt
#### Optional Arguments
     new_negative
     new_style
     new_height_width
     new_lora
     model_name
### Redraw
Redraw will generate an image using img2img to redraw an image.
#### Usage
    /redraw
#### Required Arguments
    image
    new_prompt
#### Optional Arguments
    new_negative
    new_style
    new_height_width
    new_lora
    model_name
### Crazy
Crazy will generate 4 images using a random subject, activity, and location with a random style.
#### Usage
    /crazy
### Interpret
Interpret will generate an image using a Song and Artist name to gather the lyrics, that then get sent to ChatGPT to form a prompt.
#### Usage
    /interpret
#### Required Arguments
    song_name
    artist_name
#### Optional Arguments
    new_negative
    new_style
    new_height_width
    new_lora
    model_name
### Music
Music will generate an image by the song name, artist name, and 3 random lines from the lyrics.  This will use the default style and size.  This will likely be removed at some point.
#### Usage
    /music
#### Required Arguments
    song_name
    artist_name
#### Optional Arguments
    model_name
## Installation

### Clone the repository.

    git clone https://github.com/comfyanonymous/ComfyUI.git

I recommend using a virtual environment.  This can be done by running the following commands in the root directory of the bot;

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
Edit the .env file and fill in the variables.  You will need to create a discord bot and get the token from the discord developer portal. Then fill in the Folder path with the path leading to your ComfyUI/models folder.

- `TOKEN` - Your bot's token
- `folder_path` - The path where your ComfyUI folder is located. Ex; /home/USER/ComfyUI/
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