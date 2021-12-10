import telebot

bot = telebot.TeleBot('5042122150:AAEA76aSqQBTFaIjGnQMssIpsKkGKuhQTVg')
owner_id = 384881851

@bot.message_handler(commands=['help', 'ping'])
def get_help_messages(message):
    if message.text == "/help":
        bot.send_message(message.from_user.id, "Hello, world. Запуляй в меня картинку и я найду на ней всех кожаных мешков!")
    if message.text == "/ping":
        bot.send_message(message.from_user.id, "Я живой, не пингуй")

@bot.message_handler(content_types=['alive?'])
def get_text_messages(message):
    if message.text == "Привет":
        pass

@bot.message_handler(content_types=['photo'])
def get_photo_messages(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    if (message.chat.id != owner_id):
        bot.send_message(owner_id, 'пользовательская активность:' + str(message.chat.id))
        bot.send_photo(owner_id, downloaded_file)
    #with open("botmage.jpg", 'wb') as new_file:
    #    new_file.write(downloaded_file)
    image = operateImage(downloaded_file)
    
    bot.send_message(message.chat.id, 'Вот твои кожаные мешки')
    bot.send_photo(message.chat.id, image)

@bot.message_handler(content_types=['document'])
def get_photo_doc_messages(message):
    try:
        fileID = message.document.file_id
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)

        if (message.chat.id != owner_id):
            bot.send_message(owner_id, 'пользовательская активность:' + str(message.chat.id))
            bot.send_photo(owner_id, downloaded_file)
        #with open("botmage.jpg", 'wb') as new_file:
        #    new_file.write(downloaded_file)
        image = operateImage(downloaded_file)
    
        bot.send_message(message.chat.id, 'Вот твои кожаные мешки')
        bot.send_photo(message.chat.id, image)
    except:
        pass


def operateImage(image):
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        clearPred = model([data.ImToTen(image).to(device)])[0]

    labelizated_photo = utils.labelization(image, clearPred, threshold)
    labelizated_photo = cv2.imencode('.JPEG', labelizated_photo)
    return labelizated_photo[1].tobytes()



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

import torch
import torch.nn as nn
import torch.optim as optim
import random

import torchvision
import torchvision.utils
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dataset as data
import utils as utils
import pickle
import attackMethods as am


device = torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model = model.float().to(device)

threshold = 0.8

print("Okey, im ready...")


bot.polling(none_stop=True, interval=0)