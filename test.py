import json
import websocket
import random
import asyncio
import discord
import magic
from discord.ext import commands
from discord import option
import os
from PIL import Image
import io
import tempfile
import comfyAPI
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
img2img_prompt = comfyAPI.img2img_prompt
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)
bot.auto_sync_commands = True
magic_instance = magic.Magic()


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')


with open('prompts.json', 'r') as sdxl_prompts:
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
    {"height": 1024, "width": 1024, "aspect_ratio": 1},
    {"height": 1152, "width": 896, "aspect_ratio": 1.2857142857142858},
    {"height": 896, "width": 1152, "aspect_ratio": 0.7777777777777778},
    {"height": 1216, "width": 832, "aspect_ratio": 1.4615384615384615},
    {"height": 832, "width": 1216, "aspect_ratio": 0.6842105263157895},
    {"height": 1344, "width": 768, "aspect_ratio": 1.75},
    {"height": 768, "width": 1344, "aspect_ratio": 0.5714285714285714},
    {"height": 1536, "width": 640, "aspect_ratio": 2.4},
    {"height": 640, "width": 1536, "aspect_ratio": 0.4166666666666667},
]


def get_loras():
    for dirpath, dirnames, filenames in os.walk(folder_path + '/models/'):
        subfolder_name = 'loras'
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            matching_files = [os.path.splitext(loras)[0] for loras in subfolder_files]
            return sorted(matching_files)
    # If the target subfolder is not found
    return []


async def loras_autocomplete(ctx: discord.AutocompleteContext):
    loras = get_loras()
    return [lora for lora in loras if lora.startswith(ctx.value.lower())]


async def height_width_autocomplete(ctx: discord.AutocompleteContext):
    return [f"{hw['height']} {hw['width']}" for hw in height_width_option]


async def models_autocomplete(ctx: discord.AutocompleteContext):
    subfolder_name = 'checkpoints'
    # Walk through the directory tree rooted at root_folder
    for dirpath, dirnames, filenames in os.walk(folder_path + '/models/'):
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            matching_files = [models for models in subfolder_files if models.startswith(ctx.value.lower())]
            return sorted(matching_files)

    # If the target subfolder is not found
    return []


request_queue = asyncio.Queue()
processing_lock = asyncio.Lock()


async def queue_position(ctx):
    position = request_queue.qsize()  # Get the current queue size (position in the queue)
    if position == 0:
        await ctx.send(f"Generating image for {ctx.author.mention}")
        return
    await ctx.send(f"You are in position {position} in the queue.")


async def process_requests():
    while True:
        server, author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name, includes_image = await request_queue.get()  # Dequeue the next request
        async with processing_lock:  # Acquire the lock before processing
            await process_request(server, author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name,
                                  includes_image)
            request_queue.task_done()  # Mark the request as done


async def process_request(server, author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name, includes_image):
    try:
        if includes_image:
            message, file_list = None, None
        else:
            message, file_list = await generate_image(author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name,
                                             includes_image)
        await bot.get_guild(server).text_channels[0].send(message, files=file_list)
    except Exception as e:
        print(e)
        await bot.get_guild(server).text_channels[0].send(author_name + " Something went wrong. Please try again.")


@bot.event
async def on_ready():
    bot.loop.create_task(process_requests())


async def form_message(
        author_name: str,
        new_prompt: str,
        new_negative: str = None,
        new_style: str = None,
        new_size: str = None,
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
    if new_size is not None:
        message = message + f"\n**Height/Width:** {new_size}"
    if model_name is not None:
        message = message + f"\n**Model:** {model_name}"
    return message


async def generate_image(author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name, includes_image):
    message = await form_message(author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name)
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

    if model_name is not None:
        prompt["10"]["inputs"]["ckpt_name"] = model_name
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
        return message, file_list


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
    "new_size",
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
async def draw(ctx,
               new_prompt: str,
               new_negative: str = None,
               new_style: str = None,
               new_size: str = None,
               new_lora: str = None,
               model_name: str = None
               ):
    includes_image = False
    author_name = ctx.author.mention
    server = ctx.guild.id
    await ctx.defer()
    async with processing_lock:  # Acquire the lock
        await request_queue.put(
            (server, author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name, includes_image))


@bot.slash_command(description='Go Crazy!')
async def crazy(ctx):
    await ctx.respond("Going crazy for " + ctx.author.mention)
    author_name = ctx.author.mention

    random_subject = random.choice(prompts_data["prompts"]["subjects"])
    random_verb = random.choice(prompts_data["prompts"]["verbs"])
    random_location = random.choice(prompts_data["prompts"]["locations"])
    new_prompt = f"{random_subject} {random_verb} {random_location}"
    new_negative = None
    new_style = random.choice(style_names)
    new_size = random.choice(height_width_option)
    new_lora = random.choice(get_loras())
    model_name = None
    includes_image = False
    message = form_message(author_name, new_prompt, new_negative, new_style, new_size, new_lora, model_name)
    try:
        file_list = generate_image(ctx, new_prompt, new_negative, new_style, new_size, new_lora, model_name, includes_image)
        await ctx.send(message, files=file_list)
    except Exception as e:
        print(e)
        await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


bot.run(TOKEN)
