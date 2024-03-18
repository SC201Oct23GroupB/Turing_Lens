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
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

app = Flask(__name__)

'Store tokens here'
CHANNEL_SECRET = 'd6e0a625ea6f530fd43e3e6459597e24'
CHANNEL_ACCESS_TOKEN = 'gjl/0a99GFN1kuY1L1jtBCLrusNphO/Xw9I1DBDNZlVaxlRjrR+uSqwoBJ07YKDASeFRxDEJhG5LBoQ5w8tTFV6K97hEzoV1gM7IVgCFtaGIZqknPEmG07RNREUekR0Xpu9Is5DGmZs2sBqb1Ny/EwdB04t89/1O/w1cDnyilFU='

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


@app.route("/")
def test():
    return "test OK"


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


@handler.add(MessageEvent, message=TextMessageContent)
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