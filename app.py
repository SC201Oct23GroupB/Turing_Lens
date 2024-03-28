from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageMessage, VideoMessage, AudioMessage
)

import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import json
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Tokens are stored here
CHANNEL_SECRET = 'd6e0a625ea6f530fd43e3e6459597e24'
CHANNEL_ACCESS_TOKEN = 'gjl/0a99GFN1kuY1L1jtBCLrusNphO/Xw9I1DBDNZlVaxlRjrR+uSqwoBJ07YKDASeFRxDEJhG5LBoQ5w8tTFV6K97hEzoV1gM7IVgCFtaGIZqknPEmG07RNREUekR0Xpu9Is5DGmZs2sBqb1Ny/EwdB04t89/1O/w1cDnyilFU='

# Define image preprocessing function
SIZE = 448

# Define the threshold for replies
UPPER = 60
LOWER = 40


def predict(image):
    print('Activate transform')
    transform = T.Compose([
        T.Resize((SIZE, SIZE)),
        T.ToTensor()
    ])
    print('Transform activated')

    print('Send in image to transform')
    img_trans = transform(image).unsqueeze(0)
    print('Image transform completed')

    print('Activate model')
    model = resnet50(num_classes=4)
    model.load_state_dict(torch.load("resnet50_finetuned_weights_0325_6_800.pth", map_location=torch.device('cpu')))
    model.eval()

    print('Start prediction')
    # prediction = None

    with torch.no_grad():
        output = model(img_trans)
        prob = torch.softmax(output, dim=1)
        prob = prob.squeeze().numpy()

        prob_dict = {
            'portrait': prob[0],
            'Midjourney': prob[1],
            'Stable Diffusion': prob[2],
            'Bing': prob[3]
        }

        # prediction = output.max(1)[1].item()
    portrait_pct = round(prob_dict['portrait'] * 100)
    if portrait_pct >= UPPER:
        ans = f"We are confident that this is a true portrait with {portrait_pct}% certainty! 🖼️"
    elif LOWER <= portrait_pct < UPPER:
        ans = "We recommend validating with caution, as we are not entirely certain if this is a true portrait. 🧐"
    else:
        ans = f"We are {100 - portrait_pct}% confident that this is an AI-generated portrait. 🤖"

    # map_dict = {0: 'portrait', 1: 'Midjourney', 2: 'Stable Diffusion', 3: 'Bing'}
    # ans = f"This is made by: {map_dict[prediction]}"

    return ans


@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    try:
        access_token = CHANNEL_ACCESS_TOKEN
        secret = CHANNEL_SECRET
        json_data = json.loads(body)
        print(json_data)
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        if json_data['events'][0]['message']['type'] == 'image':
            image_handler(json_data, line_bot_api)
        else:
            text_handler(json_data, line_bot_api)
    except:
        print(request.args)
    return 'OK'


# Reply for Image Message here
def image_handler(json_data, line_bot_api):
    msg_id = json_data['events'][0]['message']['id']
    print(msg_id)
    img = line_bot_api.get_message_content(msg_id).content
    print(img)
    reply_token = json_data['events'][0]['replyToken']
    print(reply_token)
    image = Image.open(BytesIO(img))
    print(image)

    try:
        prediction = predict(image)
    except:
        prediction = 'error'
        print(request.args)

    line_bot_api.reply_message(reply_token, TextSendMessage(prediction))


# Shows standard reply for message other than images
def text_handler(json_data, line_bot_api):
    reply_message = 'My superpower only works on portrait photos! Send it my way and I\'ll tell you if it\'s a work of art or AI magic! 🪄'
    reply_token = json_data['events'][0]['replyToken']
    line_bot_api.reply_message(reply_token, TextSendMessage(reply_message))


if __name__ == "__main__":
    app.run()