from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)

import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Tokens are stored here
CHANNEL_SECRET = 'd6e0a625ea6f530fd43e3e6459597e24'
CHANNEL_ACCESS_TOKEN = 'gjl/0a99GFN1kuY1L1jtBCLrusNphO/Xw9I1DBDNZlVaxlRjrR+uSqwoBJ07YKDASeFRxDEJhG5LBoQ5w8tTFV6K97hEzoV1gM7IVgCFtaGIZqknPEmG07RNREUekR0Xpu9Is5DGmZs2sBqb1Ny/EwdB04t89/1O/w1cDnyilFU='

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

model = resnet50(num_classes=4)
model.load_state_dict(torch.load("resnet50_finetuned_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing function
SIZE = 448


def preprocess(image):
    transform = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch


# Function to predict whether image is AI-generated

def predict_image(input_image):
    image = Image.fromarray(input_image.astype('uint8'), 'RGB') # 将NumPy数组转换为PIL图像
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        probabilities = probabilities.squeeze().numpy()

        probabilities_dict = {
            'portrait':probabilities[0],
            'Midjourney':probabilities[1],
            'Stable Diffusion':probabilities[2],
            'Bing':probabilities[3],
        }
        predictions=output.max(1)[1].item()
    map_dict = {0:'portrait', 1:'Midjourney', 2:'Stable Diffusion', 3:'Bing'}
    ans = f"This is made by: {map_dict[predictions]}"

    return ans


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


# Reply for Image Message here
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(message_id=event.message.id)
        image = Image.open(io.BytesIO(message_content))
        image_np = np.array(image)
        reply_message = predict_image(image_np)

    reply_message = 'Under construction'
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)]
            )
        )


# Shows standard reply for message other than images
@handler.add(MessageEvent)
def handle_message(event):
    reply_message = 'My superpower only works on portrait photos! Send it my way and I\'ll tell you if it\'s a work of art or AI magic! 🪄'
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_message)]
            )
        )


if __name__ == "__main__":
    app.run()