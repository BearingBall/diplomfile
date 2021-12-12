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
    image = operate_image(downloaded_file)
    
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
        image = operate_image(downloaded_file)
    
        bot.send_message(message.chat.id, 'Вот твои кожаные мешки')
        bot.send_photo(message.chat.id, image)
    except:
        pass


def operate_image(image):
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    with torch.no_grad():
        predict = model([data.image_to_tensor(image).to(device)])[0]

    result = utils.labelization(image, predict, threshold)
    result = cv2.imencode('.JPEG', result)
    return result[1].tobytes()




import numpy as np
import cv2 as cv2

import torch
import torchvision.utils

import data.dataset as data
import data.utils as utils

device = torch.device("cpu")

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
model = model.float().to(device)

threshold = 0.8

print("Okey, im ready...")
bot.polling(none_stop=True, interval=0)