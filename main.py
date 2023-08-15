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

load_dotenv()
TOKEN = os.getenv('TOKEN')
folder_path = os.getenv('FOLDER_PATH')
base_model = 'sd_xl_base_1.0_0.9vae.safetensors'
refiner_model = 'sd_xl_refiner_1.0_0.9vae.safetensors'

prompt = comfyAPI.prompt
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)
bot.auto_sync_commands = True

prompt["10"]["inputs"]["ckpt_name"] = base_model
prompt["4"]["inputs"]["ckpt_name"] = refiner_model

#  Height and Width options
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


upscale_option = [
    "2x",
    "4x"
]

#  Style Json parse
with open("sdxl_styles.json", 'r') as sdxl_styles:
    data = json.load(sdxl_styles)

# Parse Style names from sd_xl_styles.json
style_names = [entry["name"] for entry in data]

#  Prompts Json parse
with open("prompts.json", 'r') as sdxl_prompts:
    prompts_data = json.load(sdxl_prompts)

example_subjects = prompts_data["prompts"]["subjects"]
example_verbs = prompts_data["prompts"]["verbs"]
example_locations = prompts_data["prompts"]["locations"]


async def style_autocomplete(ctx: discord.AutocompleteContext):
    return [name for name in style_names if name.startswith(ctx.value.lower())]


async def height_width_autocomplete(ctx: discord.AutocompleteContext):
    return [height_width for height_width in height_width_option]


async def upscale_autocomplete(ctx: discord.AutocompleteContext):
    return [upscale for upscale in upscale_option]


async def read_files_in_subfolder(ctx, folder_path, subfolder_name):
    subfolder_path = os.path.join(folder_path, subfolder_name)
    if not os.path.isdir(subfolder_path):
        await ctx.send(f"Subfolder '{subfolder_name}' does not exist.")
        return

    extensions = ['.ckpt', '.pth', '.safetensors']

    for root, dirs, files in os.walk(subfolder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            _, file_extension = os.path.splitext(file_name)
            if file_extension in extensions:
                await ctx.send(file_name)


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')


@bot.slash_command(description='Generate random images with a random style')
async def crazy(ctx):
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)  # set seed for base model
    prompt["23"]["inputs"]["noise_seed"] = int(seed)  # set seed for refiner model

    # Random prompt
    # Random subject
    random_subject = random.choice(example_subjects)
    # Random verb
    random_verb = random.choice(example_verbs)
    # Random location
    random_location = random.choice(example_locations)
    new_prompt = f"{random_subject} {random_verb} {random_location}"
    prompt["146"]["inputs"]["text_positive"] = new_prompt

    # Random style
    random_entry = random.choice(data)
    random_style = random_entry["name"]
    prompt["146"]["inputs"]["style"] = random_style
    await ctx.respond(
        f"Generating 'crazy' images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {random_style}")

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        await ctx.send(
            f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {random_style}",
            files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


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
    "new_upscale",
    description="Upscale the image",
    autocomplete=upscale_autocomplete,
    required=False
)
async def draw(ctx, new_prompt: str, new_style: str, new_height_width: str, new_upscale: str):
    if new_style is not None and new_height_width is not None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {new_style}\n**Height/Width:** {new_height_width}")
    elif new_style is not None and new_height_width is None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {new_style}")
    elif new_style is None and new_height_width is not None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Height/Width:** {new_height_width}")
    else:
        await ctx.respond(f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}")

    prompt["146"]["inputs"]["text_positive"] = new_prompt
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)
    prompt["23"]["inputs"]["noise_seed"] = int(seed)
    if new_style is not None:
        if new_style == 'random':
            random_entry = random.choice(data)
            new_style = random_entry["name"]
        prompt["146"]["inputs"]["style"] = new_style
    else:
        prompt["146"]["inputs"]["style"] = 'base'
    if new_height_width:
        height, width = new_height_width.split()
        prompt["5"]["inputs"]["height"] = int(height)
        prompt["5"]["inputs"]["width"] = int(width)
    else:
        prompt["5"]["inputs"]["height"] = 1024
        prompt["5"]["inputs"]["width"] = 1024
    if new_upscale:
        if new_upscale == "2x":
            prompt["148"]["inputs"]["scale_by"] = 2
            print("Upscaling by 2x")
        elif new_upscale == "4x":
            prompt["148"]["inputs"]["scale_by"] = 4
            print("Upscaling by 4x")
    else:
        prompt["148"]["inputs"]["scale_by"] = 1
        print("Upscaling by 1x")
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        if new_style is not None and new_height_width is not None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {new_style}\n**Height/Width:** {new_height_width}",
                files=file_list)
        elif new_style is not None and new_height_width is None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {new_style}",
                files=file_list)
        elif new_style is None and new_height_width is not None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Height/Width:** {new_height_width}",
                files=file_list)
        else:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}",
                files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


bot.run(TOKEN)
