import logging
import os
import shutil
from keras.models import load_model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import asyncio
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

#включаем логирование
logging.basicConfig(level=logging.INFO)
#объект бота
TOKEN = '6888603985:AAHiPTXFOB4HFjADgk4bPpMV-EP6MYoARMM'
#диспетчер
dp = Dispatcher()
#обработка команды старт
@dp.message(CommandStart())
async def echo(message: Message):
  await message.reply('Привет! Я бот, распознающий овощи и фрукты')

# обработка команды help
@dp.message(F.text, Command("test"))
async def echo(message: Message):
  await message.reply('Просто отправьте мне изображение, которое содержит овощ или фрукт')


def get_img_array(img_path, size):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
  array = tf.keras.preprocessing.image.img_to_array(img)
  # расширяем размерность для преобразования массива в пакеты
  array = np.expand_dims(array, axis=0)
  return array

def predictions(img_path):
  img_size = (224, 224)
  classifier = load_model(f"{os.path.dirname(__file__)}/fruit224mobile.h5")
  class_labels = ['Яблоко', 'Банан', 'Свекла', 'Болгарский перец', 'Капуста', 'Стручковый перец', 'Морковь', 'Цветная капуста',
                'Перец чили', 'Кукуруза', 'Огурец', 'Баклажан', 'Чеснок', 'Имбирь', 'Виноград', 'Халапеньо', 'Киви',
                'Лимон', 'Латук', 'Манго', 'Лук', 'Апельсин', 'Паприка', 'Груша', 'Горох', 'Ананас', 'Гранат',
                'Картофель', 'Редька', 'Соевые бобы', 'Шпинат', 'Сладкая кукуруза', 'Батат', 'Помидор', 'Репа', 'Арбуз']
  try:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    pred = np.argmax(classifier.predict(img_array), axis=1)
    predictions = class_labels[pred[0]]
    return predictions
  except Exception:
    return "Проблемы с изображением"

@dp.message(F.text)
async def handle_message(message: Message):
  await message.reply('Просто отправьте мне изображение, которое содержит овощ или фрукт')

@dp.message(F.photo)
async def download_photo(message: Message, bot: Bot):
  folder_path = os.path.dirname(__file__) + "/photo"
  try:
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
  except:
    os.mkdir(folder_path)
  await bot.download(
      message.photo[-1],
      destination=f"{folder_path}/{message.photo[-1].file_id}.jpg"
  )
  # определяем путь к фото
  img_path = f"{folder_path}/{message.photo[-1].file_id}.jpg"
# получаем предсказание
  pred = predictions(img_path)
# # Отправляем ответ пользователю
  await message.answer(f"Я думаю, что это {pred}😊")


async def main() -> None:
  # Initialize Bot instance with default bot properties which will be passed to all API calls
  bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
  # And the run events dispatching
  await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())