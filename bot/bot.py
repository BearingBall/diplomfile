import telebot

bot = telebot.TeleBot('5042122150:AAEJupf6lF5aA5au2_5xe8gOG6MTakN96OM')
owner_id = 384881851


@bot.message_handler(commands=['help'])
def get_help_messages(message):
    if message.text == "/help":
        bot.send_message(message.from_user.id, "Ну просто пикчу закидывай и полетели")


@bot.message_handler(content_types=['alive?'])
def get_text_messages(message):
    if message.text == "Привет":
        pass


@bot.message_handler(content_types=['photo'])
def get_photo_messages(message):
    bot.send_message(message.chat.id, 'Ща, пагодь')

    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    if (message.chat.id != owner_id):
        bot.send_message(owner_id, 'пользовательская активность:' + str(message.chat.id))
        bot.send_photo(owner_id, downloaded_file)
    #with open("botmage.jpg", 'wb') as new_file:
    #    new_file.write(downloaded_file)
    image, image_attacked = operate_image(downloaded_file)

    bot.send_message(message.chat.id, 'В сыром виде:')
    bot.send_photo(message.chat.id, image)
    bot.send_message(message.chat.id, 'Спрятал:')
    bot.send_photo(message.chat.id, image_attacked)


@bot.message_handler(content_types=['document'])
def get_photo_doc_messages(message):
    try:
        bot.send_message(message.chat.id, 'Ща, пагодь')

        fileID = message.document.file_id
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)

        if (message.chat.id != owner_id):
            bot.send_message(owner_id, 'пользовательская активность:' + str(message.chat.id))
            bot.send_photo(owner_id, downloaded_file)
        #with open("botmage.jpg", 'wb') as new_file:
        #    new_file.write(downloaded_file)
        image, image_attacked = operate_image(downloaded_file)
    
        bot.send_message(message.chat.id, 'В сыром виде:')
        bot.send_photo(message.chat.id, image)
        bot.send_message(message.chat.id, 'Спрятал:')
        bot.send_photo(message.chat.id, image_attacked)

    except:
        bot.send_message(message.chat.id, 'Чет не поперло')
        pass


def operate_image(image):
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = utils.image_to_tensor(image).to(device)

    with torch.no_grad():
        predict = model([image])[0]
    result_raw = attack_utils.visualize_labels_predicted(utils.tensor_to_image(image), predict, threshold)

    image_attacked = image

    for i in range(len(predict["labels"])):
        if predict["scores"][i] < threshold:
            continue

        if predict["labels"][i] == 1:
            box = [int(predict['boxes'][i][0]), 
                    int(predict['boxes'][i][1]), 
                    int(predict['boxes'][i][2] - predict['boxes'][i][0]),
                    int(predict['boxes'][i][3] - predict['boxes'][i][1])]
            
            image_attacked = attack.insert_patch(image_attacked, patch, box, 0.4, device, True)


    with torch.no_grad():
        predict = model([image_attacked])[0]        
    result_attacked = attack_utils.visualize_labels_predicted(utils.tensor_to_image(image_attacked), predict, threshold)

    result_raw = cv2.cvtColor(result_raw, cv2.COLOR_RGB2BGR)
    result_attacked = cv2.cvtColor(result_attacked, cv2.COLOR_RGB2BGR)

    return cv2.imencode('.JPEG', result_raw)[1].tobytes(), cv2.imencode('.JPEG', result_attacked)[1].tobytes()


def predict_image(image):
    with torch.no_grad():
        predict = model([image])[0]

    return utils.labelization(image, predict, threshold)


import sys
from pathlib import Path
sys.path.append(Path(sys.path[0]).parent.as_posix())

import numpy as np
import cv2 as cv2

import torch
import torchvision.utils

import data.utils as utils
import attack_construction.attack_methods as attack
import attack_construction.utils as attack_utils

device = torch.device("cpu")

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
model = model.float().to(device)

threshold = 0.4

patch = cv2.cvtColor(cv2.imread('bot\patch.png'), cv2.COLOR_BGR2RGB)
patch = utils.image_to_tensor(patch)

print("Okey, im ready...")
bot.polling(none_stop=True, interval=0)