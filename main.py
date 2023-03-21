from model import StyleTransferModel
from telegram_token import token
from io import BytesIO
from config import start, help
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import logging

model = StyleTransferModel()
first_image_file = {}

async def send_prediction_on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = context.bot.get_file(image_info)

    if chat_id in first_image_file:

        # первая картинка, которая к нам пришла станет content image, а вторая style image
        content_image_stream = BytesIO()
        first_image_file[chat_id].download(out=content_image_stream)
        del first_image_file[chat_id]

        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)
        #input_img = content_image_stream.clone()
        #input_img = torch.randn(content_img.data.size(), device=device)

        output = model.transfer_style(content_img=content_image_stream, style_img=style_image_stream)

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        await context.bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
    else:
        first_image_file[chat_id] = image_file


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help))
    #TODO: доделать приём и обработку изображений для асинхронной версии
#    app.add_handler(MessageHandler(MessageHandler.filters.attached, ))
    app.run_polling()




'''    updater = Updater(token=token,  request_kwargs={'proxy_url': 'socks4://23.252.66.25:54321'})
    updater.dispatcher.add_handler(MessageHandler(MessageHandler.filters.photo, send_prediction_on_photo))'''