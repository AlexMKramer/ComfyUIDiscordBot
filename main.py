import json
import asyncio
import csv
import PIL.Image
import websocket
import random
import discord
import magic
from discord.ext import commands, tasks
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
genius = Genius(os.getenv('GENIUS_TOKEN'))
openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = comfyAPI.prompt
img2img_prompt = comfyAPI.img2img_prompt
upscale_prompt = comfyAPI.upscale_prompt

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)
bot.auto_sync_commands = True

magic_instance = magic.Magic()
wait_message = []


def random_message():
    with open('resources/messages.csv', encoding='UTF-8') as csv_file:
        message_data = list(csv.reader(csv_file, delimiter='|'))
        for row in message_data:
            wait_message.append(row[0])
    wait_message_count = len(wait_message) - 1
    text = wait_message[random.randint(0, wait_message_count)]
    return text


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')
    # Start the image queue
    image_queue.start()


'''    bot.loop.create_task(process_commands())'''


@bot.event
async def on_disconnect():
    print(f'Disconnected from {bot.user.name}')
    await bot.connect(reconnect=True)


command_queue = asyncio.Queue()
queue_processing = False


# Function to check if an image is in the queue and/or processing and return what the queue placement is
def check_queue_placement():
    # Check if the queue has something in it
    if not command_queue.empty():
        # Check if the queue is processing
        if queue_processing:
            queue_spot = command_queue.qsize() + 1
            return queue_spot
        else:
            queue_spot = command_queue.qsize()
            return queue_spot
    else:
        if queue_processing:
            return 1
        else:
            return 0


# Image queue loop to process images
@tasks.loop(seconds=1)
async def image_queue():
    global queue_processing
    if not command_queue.empty():
        try:
            command = await command_queue.get()
            queue_processing = True
            channel_id, author_name, message, is_img2img, new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name = command
            channel = bot.get_channel(channel_id)
            print(f'Generating image {command}')
            loop = asyncio.get_event_loop()
            try:
                if is_img2img:
                    await channel.send("**" + random_message() + "**" + "\nRecreating image...")
                    file_list = await loop.run_in_executor(None, generate_img2img, new_prompt, percent_of_original,
                                                           new_negative, new_style, new_size, new_lora, lora_strength,
                                                           artist_name, model_name)
                else:
                    await channel.send("**" + random_message() + "**" + "\nGenerating images...")
                    file_list = await loop.run_in_executor(None, generate_image, new_prompt, new_negative, new_style,
                                                           new_size, new_lora, lora_strength, artist_name, model_name)
                await channel.send(message, files=file_list)
                queue_processing = False
            except Exception as e:
                print(e)
                await channel.send(author_name + " Something went wrong. Please try again.")
        except Exception as e:
            print(f'Error processing image: {e}')


def gpt_integration(text):
    gpt_new_prompt = ({"role": "user", "content": "Here are the lyrics I would like in this format:" + text})
    gpt_message = gpt_initial_prompt + [gpt_new_prompt]
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_message
        )
        reply_content = completion.choices[0].message.content
        start_index = reply_content.find('{')
        end_index = reply_content.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            new_text = reply_content[start_index:end_index + 1]
            print(f'ChatGPT reply: {new_text}')
            return new_text
        else:
            print(f'ChatGPT reply: {reply_content}')
            return reply_content
    except Exception as e:
        print(e)
        return None


gpt_initial_prompt = [{'role': 'user',
                       'content': "Using song lyrics, come up with a prompt for an image generator.  "
                                  "Please follow the format exactly. The format should be broken down "
                                  "like this: {Art Style}, {Subject}, {Details}, {Color}\n The art style "
                                  "should be determined by the overall impression of the song.  If it is "
                                  "sad, then something like La Douleur should be used. If it is happy, "
                                  "perhaps a vibrant street art style.\nThe Subject should be determined "
                                  "by who the song is about.  If the song is about a couple trying to "
                                  "escape the city, then the subject should be a couple.\nThe Details "
                                  "should be determined by descriptive words used in the song.  If they "
                                  "mention empty bottles, then add empty bottles to the prompt.\nThe "
                                  "color should be determined by the mood of the song.  If the mood is a "
                                  "happy one, use bright colors.\nHere is an example:\n{A dreamlike and "
                                  "ethereal art style}, {a couple standing on a cliffside embracing, "
                                  "overlooking a surreal and beautiful landscape}, {sunset, grassy, "
                                  "soft wind}, {soft pastels, with hints of warm oranges and pinks}"},
                      {'role': 'assistant',
                       'content': "{Vibrant and energetic street art style}, {a group of friends dancing and "
                                  "celebrating under the city lights}, {joyful, urban, rhythm}, {bold and lively "
                                  "colors, with splashes of neon blues and pinks}"}, ]

with open('resources/prompts.json', 'r') as sdxl_prompts:
    prompts_data = json.load(sdxl_prompts)

example_subjects = prompts_data["prompts"]["subjects"]
example_verbs = prompts_data["prompts"]["verbs"]
example_locations = prompts_data["prompts"]["locations"]

with open("resources/sdxl_styles.json", 'r') as sdxl_styles:
    data = json.load(sdxl_styles)
# Parse Style names from sdxl_styles.json
style_names = [entry["name"] for entry in data]


async def style_autocomplete(ctx: discord.AutocompleteContext):
    return [name for name in style_names if name.startswith(ctx.value.lower())]


with open("resources/artists.json", 'r') as sdxl_artists:
    artists = json.load(sdxl_artists)
# Parse Style names from artists.json
artist_names = [entry["name"] for entry in artists]


async def artist_autocomplete(ctx: discord.AutocompleteContext):
    return [name for name in artist_names if name.startswith(ctx.value.title())]


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


def fix_lyrics(text):
    keyword1 = "Lyrics"
    keyword2 = r"\d*Embed|Embed"
    start_index = text.find(keyword1)
    end_index = re.search(keyword2, text[start_index])
    try:
        if start_index != -1 and end_index:
            lyrics_in_index = start_index + end_index.start()
            text = text[start_index + len(keyword1):lyrics_in_index].strip()
        else:
            text = text
        ad_pattern = r'See .*? LiveGet tickets as low as \$\d+You might also like'
        re.sub(ad_pattern, '', text)
        re.sub(r'\[.*?\]', '', text)
        re.sub(r'\d+$', '', text)
        re.sub('"', '', text)
    except Exception as e:
        print(f'Error: {e}\'\nLyrics: {text}')
    return text


def get_lyrics(song, artist):
    try:
        song = genius.search_song(song, artist)
        new_lyrics = song.lyrics
        fixed_lyrics = fix_lyrics(new_lyrics)
        return fixed_lyrics
    except Exception as e:
        print(e)
        return None


def form_message(
        author_name: str,
        new_prompt: str,
        percent_of_original: int = None,
        new_negative: str = None,
        new_style: str = None,
        new_size: str = None,
        new_lora: str = None,
        lora_strength: int = None,
        artist_name: str = None,
        model_name: str = None
):
    message = f"Generated images for {author_name}\n**Prompt:** {new_prompt}"
    if percent_of_original is not None:
        message = message + f"\n**Percent of Original:** {percent_of_original}%"
    if new_negative is not None:
        message = message + f"\n**Negative Prompt:** {new_negative}"
    if new_lora is not None:
        message = message + f"\n**Lora:** {new_lora}"
        if lora_strength is not None:
            message = message + f"\n**Lora Strength:** {lora_strength}"
    if new_style is not None:
        message = message + f"\n**Style:** {new_style}"
    if new_size is not None:
        message = message + f"\n**Height/Width:** {new_size}"
    if artist_name is not None:
        message = message + f"\n**Artist:** {artist_name}"
    if model_name is not None:
        message = message + f"\n**Model:** {model_name}"
    return message


def generate_image(new_prompt, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name):
    if new_lora is not None:
        if lora_strength is None:
            lora_strength = 0.5
        else:
            lora_strength = lora_strength / 10
        new_prompt = " <lora:" + new_lora + ":" + str(lora_strength) + ">, " + new_prompt
    if artist_name is not None:
        new_prompt = new_prompt + ", by " + artist_name
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
        return file_list


def generate_img2img(new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora, lora_strength,
                     artist_name, model_name):
    if new_lora is not None:
        if lora_strength is None:
            lora_strength = 0.5
        else:
            lora_strength = lora_strength / 10
        new_prompt = " <lora:" + new_lora + ":" + str(lora_strength) + ">, " + new_prompt
    if artist_name is not None:
        new_prompt = new_prompt + ", by " + artist_name
    img2img_prompt["146"]["inputs"]["text_positive"] = new_prompt
    if percent_of_original >= 85:
        percent_of_original = 84
    percent_of_original = percent_of_original / 2
    img2img_prompt["22"]["inputs"]["start_at_step"] = int(percent_of_original)
    if new_negative is not None:
        img2img_prompt["146"]["inputs"]["text_negative"] = new_negative
    else:
        img2img_prompt["146"]["inputs"]["text_negative"] = ''

    if new_style is not None:
        if new_style == 'random':
            new_style = random.choice(style_names)
        img2img_prompt["146"]["inputs"]["style"] = new_style
    else:
        img2img_prompt["146"]["inputs"]["style"] = 'base'

    if model_name is not None:
        img2img_prompt["10"]["inputs"]["ckpt_name"] = model_name
    else:
        img2img_prompt["10"]["inputs"]["ckpt_name"] = base_model

    if new_size is not None:
        height, width = new_size.split()
        img2img_prompt["160"]["inputs"]["height"] = int(height)
        img2img_prompt["160"]["inputs"]["width"] = int(width)
    else:
        img2img_prompt["160"]["inputs"]["height"] = 1024
        img2img_prompt["160"]["inputs"]["width"] = 1024

    seed = random.randint(0, 0xffffffffff)
    img2img_prompt["22"]["inputs"]["noise_seed"] = int(seed)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    images = comfyAPI.get_images(ws, img2img_prompt)
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


def generate_upscale():
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    images = comfyAPI.get_images(ws, upscale_prompt)
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
    "lora_strength",
    description="Strength 1-10",
    min_value=1,
    max_value=10,
    required=False
)
@option(
    "artist_name",
    description="Choose an artist name",
    autocomplete=artist_autocomplete,
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
               lora_strength: int = None,
               artist_name: str = None,
               model_name: str = None
               ):
    print(f'Draw Command received: {ctx}')
    # Setup message
    author_name = ctx.author.mention
    percent_of_original = None
    global queue_processing
    message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora,
                           lora_strength, artist_name, model_name)
    if check_queue_placement() != 0:
        await ctx.respond(f"You are number {check_queue_placement()} in the queue. Please wait patiently.")
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))
    else:
        await ctx.defer(invisible=True)
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))

        # try:
    #     file_list = generate_image(new_prompt, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name)
    #     await ctx.send(message, files=file_list)
    # except Exception as e:
    #     print(e)
    #     await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


@bot.slash_command(description='Go Crazy!')
async def crazy(ctx):
    author_name = ctx.author.mention
    new_negative = percent_of_original = new_lora = lora_strength = model_name = None
    random_subject = random.choice(prompts_data["prompts"]["subjects"])
    random_verb = random.choice(prompts_data["prompts"]["verbs"])
    random_location = random.choice(prompts_data["prompts"]["locations"])
    artist_name = random.choice(artist_names)
    new_prompt = f"{random_subject} {random_verb} {random_location}"
    new_style = random.choice(style_names)
    random_size = random.choice(height_width_option)
    new_size = str(random_size["height"]) + " " + str(random_size["width"])
    message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora,
                           lora_strength, artist_name, model_name)
    if check_queue_placement() != 0:
        await ctx.respond(f"You are number {check_queue_placement()} in the queue. Please wait patiently.")
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))
    else:
        await ctx.respond("**" + random_message() + "**" + "\nGoing crazy for " + ctx.author.mention)
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))

    # try:
    #     file_list = generate_image(new_prompt, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name)
    #     await ctx.send(message, files=file_list)
    # except Exception as e:
    #     print(e)
    #     await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


@bot.slash_command(description="Interpret a song's lyrics using ChatGPT!")
@option(
    "song",
    description="Enter the song name",
    required=True
)
@option(
    "artist",
    description="Enter the artist name",
    required=True
)
@option(
    "model_name",
    description="Choose the model",
    autocomplete=models_autocomplete,
    required=False
)
async def interpret(ctx,
                    song: str,
                    artist: str,
                    model_name: str = None
                    ):
    author_name = ctx.author.mention
    new_negative = new_style = new_lora = lora_strength = artist_name = percent_of_original = None
    new_size = "1344 768"
    if check_queue_placement() != 0:
        await ctx.respond(f"You are number {check_queue_placement()} in the queue. Please wait patiently.")
        fixed_lyrics = get_lyrics(song, artist)
        if fixed_lyrics is None:
            await ctx.send("Lyrics not found. Please check your spelling try again.")
            return
        await ctx.send("Interpreting lyrics...")
        new_prompt = gpt_integration(fixed_lyrics)
        if new_prompt is None:
            await ctx.send("Something went wrong. Please try again.")
            return
        message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size,
                               new_lora,
                               lora_strength, artist_name, model_name)
        await command_queue.put((
            ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
            new_lora, lora_strength, artist_name, model_name))

    else:
        await ctx.respond("**" + random_message() + "**" + f"\nGetting lyrics:\n**Song:** {song}\n**Artist:** {artist}")
        fixed_lyrics = get_lyrics(song, artist)
        if fixed_lyrics is None:
            await ctx.send("Lyrics not found. Please check your spelling try again.")
            return
        await ctx.send("Interpreting lyrics...")
        new_prompt = gpt_integration(fixed_lyrics)
        if new_prompt is None:
            await ctx.send("Something went wrong. Please try again.")
            return
        message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size,
                               new_lora,
                               lora_strength, artist_name, model_name)
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))
    # try:
    #     file_list = generate_image(new_prompt, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name)
    #     await ctx.send(message, files=file_list)
    # except Exception as e:
    #     print(e)
    #     await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


@bot.slash_command(description="Generate images based on song lyrics!")
@option(
    "song",
    description="Enter the song name",
    required=True
)
@option(
    "artist",
    description="Enter the artist name",
    required=True
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def music(ctx,
                song: str,
                artist: str,
                model_name: str = None
                ):
    author_name = ctx.author.mention
    new_negative = new_style = new_size = new_lora = lora_strength = artist_name = percent_of_original = None
    await ctx.respond(
        f"Generating images:\n**Song:** {song}\n**Artist:** {artist}")
    fixed_lyrics = get_lyrics(song, artist)
    if fixed_lyrics is None:
        await ctx.send("Lyrics not found. Please check your spelling try again.")
        return
    await ctx.send("**" + random_message() + "**" + "\nGot lyrics...")
    if check_queue_placement() != 0:
        await ctx.respond(f"You are number {check_queue_placement()} in the queue. Please wait patiently.")
        fixed_lyrics = get_lyrics(song, artist)
        if fixed_lyrics is None:
            await ctx.send("Lyrics not found. Please check your spelling try again.")
            return
        await ctx.send("Interpreting lyrics...")
        lines = fixed_lyrics.split('\n')
        lines = [line for line in lines if line.strip()]
        lines = [line for line in lines if '[' not in line and ']' not in line]
        random_lines = random.sample(lines, min(3, len(lines)))
        output_line = ', '.join(random_lines)
        new_prompt = song + ", " + output_line + ", " + artist
        if new_prompt is None:
            await ctx.send("Something went wrong. Please try again.")
            return
        message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size,
                               new_lora,
                               lora_strength, artist_name, model_name)
        await command_queue.put((
            ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
            new_lora, lora_strength, artist_name, model_name))
    else:
        await ctx.respond("**" + random_message() + "**" + f"\nGetting lyrics:\n**Song:** {song}\n**Artist:** {artist}")
        fixed_lyrics = get_lyrics(song, artist)
        if fixed_lyrics is None:
            await ctx.send("Lyrics not found. Please check your spelling try again.")
            return
        await ctx.send("Interpreting lyrics...")
        lines = fixed_lyrics.split('\n')
        lines = [line for line in lines if line.strip()]
        lines = [line for line in lines if '[' not in line and ']' not in line]
        random_lines = random.sample(lines, min(3, len(lines)))
        output_line = ', '.join(random_lines)
        new_prompt = song + ", " + output_line + ", " + artist
        if new_prompt is None:
            await ctx.send("Something went wrong. Please try again.")
            return
        message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size,
                               new_lora,
                               lora_strength, artist_name, model_name)
        await command_queue.put(
            (ctx.channel.id, author_name, message, new_prompt, percent_of_original, new_negative, new_style, new_size,
             new_lora, lora_strength, artist_name, model_name))
    # try:
    #     file_list = generate_image(new_prompt, new_negative, new_style, new_size, new_lora, lora_strength, artist_name, model_name)
    #     await ctx.send(message, files=file_list)
    # except Exception as e:
    #     print(e)
    #     await ctx.send(ctx.author.mention + " Something went wrong. Please try again.")


@bot.slash_command(description='Generate an image using an image and words!')
@option(
    "attached_image",
    description="Attach an image",
    required=True
)
@option(
    "new_prompt",
    description="Enter the prompt",
    required=True
)
@option(
    "percent_of_original",
    description="How close to the original image should the new image be?",
    required=True
)
@option(
    "new_negative",
    description="Enter things you don't want to see in the image",
    required=False
)
@option(
    "new_style",
    description="Enter the style",
    autocomplete=style_autocomplete,
    required=False
)
@option(
    "new_lora",
    description="Choose the Lora model",
    autocomplete=loras_autocomplete,
    required=False
)
@option(
    "lora_strength",
    description="Strength 1-10",
    min_value=1,
    max_value=10,
    required=False
)
@option(
    "artist_name",
    description="Choose an artist name",
    autocomplete=artist_autocomplete,
    required=False
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def redraw(ctx,
                 attached_image: discord.Attachment,
                 new_prompt: str,
                 percent_of_original: int,
                 new_negative: str = None,
                 new_style: str = None,
                 new_lora: str = None,
                 lora_strength: int = None,
                 artist_name: str = None,
                 model_name: str = None
                 ):
    author_name = ctx.author.mention
    await ctx.respond("**" + random_message() + "**" + f"\nGenerating image:\n**Prompt:** {new_prompt}")
    image_bytes = await attached_image.read()

    # Process the image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    aspect_ratio = height / width
    print(f"Image aspect ratio: {aspect_ratio}")

    closest_option = min(height_width_option, key=lambda option: abs(option["aspect_ratio"] - aspect_ratio))

    new_width = closest_option["width"]
    new_height = closest_option["height"]

    new_img = image.resize((new_width, new_height), Image.LANCZOS)
    print(f'New image size: {new_img.size}')
    image.save(folder_path + '/input/temp_image.png')
    output = io.BytesIO()
    image.save(output, format='PNG')  # You can adjust the format if needed
    output.seek(0)
    await ctx.send(f"Original image:")
    await ctx.send(file=discord.File(output, filename="original_image.png"))
    print(f'New image saved to input/temp_image.png')
    # convert height and width back to new_size string
    new_size = str(new_height) + " " + str(new_width)
    print(f'New size: {new_size}')
    message = form_message(author_name, new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora,
                           lora_strength, artist_name, model_name)
    try:
        file_list = generate_img2img(new_prompt, percent_of_original, new_negative, new_style, new_size, new_lora,
                                     lora_strength, artist_name, model_name)
        await ctx.send(message)
        await ctx.send("New image:", files=file_list)
    except Exception as e:
        print(e)
        await ctx.send(ctx.author.mention + "img2img issue.")


"""@bot.slash_command(description='Upscale an image!')
@option(
    "attached_image",
    description="Attach an image",
    required=True
)
async def upscale(ctx, attached_image: discord.Attachment):
    author_name = ctx.author.mention
    await ctx.respond(f"Upscaling image...")
    image_bytes = await attached_image.read()

    # Convert image to png
    image = Image.open(io.BytesIO(image_bytes))
    image.save(folder_path + '/input/temp_upscale.png')
    message = "Upscaled image for " + author_name + ":"
    try:
        PIL.Image.MAX_IMAGE_PIXELS = None
        file_list = generate_upscale()
        await ctx.send(message, files=file_list)
    except Exception as e:
        print(e)
        await ctx.send(ctx.author.mention + "upscale issue.")"""

if __name__ == '__main__':
    # asyncio.run(main())
    # background_task_loop.create_task(process_command())
    bot.run(TOKEN)
