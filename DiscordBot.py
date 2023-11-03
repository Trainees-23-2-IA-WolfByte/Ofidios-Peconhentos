import discord
from discord.ext import commands
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import modelo1.h5 from Modelos
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix='!', intents=intents)

model_path = r'Modelos/modelo1.h5'
token = 'MTE2OTk5NzQwNDMzNDk5MzQ3MQ.GUt73k.jRtyprxJc2TpdIFIHTjbG-rjEQs-wqvXSr-rSA'

modelo = tf.keras.models.load_model(model_path)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def check(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("Nenhuma imagem encontrada. Por favor, envie uma imagem com o comando.")
        return

    for file in ctx.message.attachments:
        if file.content_type.startswith('image'):
            image_path = file.filename
            await file.save(image_path)

            img = image.load_img(image_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255
            prediction = modelo.predict(img_array)

            if prediction < 0.5:
                await ctx.send("Não é uma serpente peçonhenta.\n")
            else:
                await ctx.send("É uma serpente peçonhenta.\n")

            os.remove(image_path)
            return

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandError):
        await ctx.send(error)

bot.run(token)

