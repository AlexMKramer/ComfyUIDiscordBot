import asyncio
import json
import queue
import websocket
import random
import discord
from discord.ext import commands
from discord import option
import os
from PIL import Image
import io
import tempfile
import comfyAPI
from typing import Optional
from dotenv import load_dotenv
import re
from lyricsgenius import Genius
import openai

load_dotenv()
TOKEN = os.getenv('TOKEN')
folder_path = os.getenv('FOLDER_PATH')
base_model = 'sd_xl_base_1.0_0.9vae.safetensors'
refiner_model = 'sd_xl_refiner_1.0_0.9vae.safetensors'
genius_token = os.getenv('GENIUS_TOKEN')
genius = Genius(genius_token)
openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = comfyAPI.prompt
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)
bot.auto_sync_commands = True


with open("prompts.json", 'r') as sdxl_prompts:
    prompts_data = json.load(sdxl_prompts)

example_subjects = prompts_data["prompts"]["subjects"]
example_verbs = prompts_data["prompts"]["verbs"]
example_locations = prompts_data["prompts"]["locations"]

with open("sdxl_styles.json", 'r') as sdxl_styles:
    data = json.load(sdxl_styles)
# Parse Style names from sd_xl_styles.json
style_names = [entry["name"] for entry in data]


async def style_autocomplete(ctx: discord.AutocompleteContext):
    return [name for name in style_names if name.startswith(ctx.value.lower())]


height_width_option = [
    "1024 1024",
    "1152 896",
    "896 1152",
    "1216 832",
    "832 1216",
    "1344 768",
    "768 1344",
    "1536 640",
    "640 1536"
]


async def height_width_autocomplete(ctx: discord.AutocompleteContext):
    return [height_width for height_width in height_width_option]


async def loras_autocomplete(ctx: discord.AutocompleteContext):
    subfolder_name = 'loras'
    # Walk through the directory tree rooted at root_folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            matching_files = [os.path.splitext(loras)[0] for loras in subfolder_files if
                              loras.startswith(ctx.value.lower())]
            return sorted(matching_files)

    # If the target subfolder is not found
    return []


async def models_autocomplete(ctx: discord.AutocompleteContext):
    subfolder_name = 'checkpoints'
    # Walk through the directory tree rooted at root_folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            matching_files = [models for models in subfolder_files if models.startswith(ctx.value.lower())]
            return sorted(matching_files)

    # If the target subfolder is not found
    return []


async def form_message(
        author_name: str,
        new_prompt: str,
        new_negative: str = None,
        new_style: str = None,
        new_height_width: str = None,
        new_lora: str = None,
        model_name: str = None
):
    message = f"Generating images for {author_name}\n**Prompt:** {new_prompt}"
    if new_negative is not None:
        message = message + f"\n**Negative Prompt:** {new_negative}"
    if new_lora is not None:
        message = message + f"\n**Lora:** {new_lora}"
    if new_style is not None:
        message = message + f"\n**Style:** {new_style}"
    if new_height_width is not None:
        message = message + f"\n**Height/Width:** {new_height_width}"
    if model_name is not None:
        message = message + f"\n**Model:** {model_name}"
    return message


def generate_image(new_prompt, new_negative, new_style, new_size, new_lora, new_model):
    if new_lora is not None:
        new_prompt = " <lora:" + new_lora + ":0.5>, " + new_prompt
    prompt["146"]["inputs"]["text_positive"] = new_prompt

    if new_negative is not None:
        prompt["146"]["inputs"]["text_negative"] = new_negative
    else:
        prompt["146"]["inputs"]["text_negative"] = ''

    if new_style is not None:
        if new_style == 'random':
            new_style = random.choice(style_names)
        prompt["146"]["inputs"]["style"] = new_style
    else:
        prompt["146"]["inputs"]["style"] = 'base'

    if new_model is not None:
        prompt["10"]["inputs"]["ckpt_name"] = new_model
    else:
        prompt["10"]["inputs"]["ckpt_name"] = base_model

    if new_size is not None:
        height, width = new_size.split()
        prompt["5"]["inputs"]["height"] = int(height)
        prompt["5"]["inputs"]["width"] = int(width)
    else:
        prompt["5"]["inputs"]["height"] = 1024
        prompt["5"]["inputs"]["width"] = 1024

    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)
    prompt["23"]["inputs"]["noise_seed"] = int(seed)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        for file_path in file_paths:
            os.remove(file_path)
        return file_list


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')


@bot.slash_command(description='Generate images using only words!')
@option(
    "new_prompt",
    description="Enter the prompt",
    required=True
)
@option(
    "new_style",
    description="Enter the style",
    autocomplete=style_autocomplete,
    required=False
)
@option(
    "new_height_width",
    description="Choose the height and width",
    autocomplete=height_width_autocomplete,
    required=False
)
@option(
    "new_lora",
    description="Choose the Lora model",
    autocomplete=loras_autocomplete,
    required=False
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def draw(
        ctx,
        new_prompt: str,
        new_negative: str = None,
        new_style: str = None,
        new_height_width: str = None,
        new_lora: str = None,
        model_name: str = None
):
    # Setup message
    author_name = ctx.author.mention
    message = form_message(author_name, new_prompt, new_negative, new_style, new_height_width, new_lora, model_name)
    await ctx.respond(message)
    try:
        file_list = generate_image(new_prompt, new_negative, new_style, new_height_width, new_lora, model_name)
        await ctx.send(message, files=file_list)
    except Exception as e:
        print(e)
        await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


@bot.slash_command(description='Go Crazy!')
async def crazy(ctx):
    author_name = ctx.author.mention

    random_subject = random.choice(prompts_data["prompts"]["subjects"])
    random_verb = random.choice(prompts_data["prompts"]["verbs"])
    random_location = random.choice(prompts_data["prompts"]["locations"])
    new_prompt = f"{random_subject} {random_verb} {random_location}"
    new_negative = None
    new_style = random.choice(style_names)
    new_height_width = random.choice(height_width_option)
    lora_results = await loras_autocomplete(ctx)
    if lora_results:
        new_lora = random.choice(lora_results)
        print("Random Lora:", new_lora)
    else:
        new_lora = None
        print("No matching loras found.")
    model_name = None
    message = form_message(author_name, new_prompt, new_negative, new_style, new_height_width, new_lora, model_name)
    try:
        file_list = generate_image(new_prompt, new_negative, new_style, new_height_width, new_lora, model_name)
        await ctx.send(message, files=file_list)
    except Exception as e:
        print(e)
        await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")

bot.run(TOKEN)
