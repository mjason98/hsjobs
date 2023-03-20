import telebot, os
from decouple import config
from my_code.models import predict

TELEGRAM_TOKEN = config('TELEGRAM_TOKEN')

bot = telebot.TeleBot(token=TELEGRAM_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    text = """
    Hi. This bot will accept the description of a Job Post and it will predict if it's a fraudulent or not.\nTo use it, just send the description as a text file.
    """
    bot.reply_to(message, text)

@bot.message_handler(func= lambda message: message.content_type == "text")
def handle_job_post(message):
    
    prediction = predict(message.text)
    if prediction == 1:
        response = "This job posting is probably a fraudulent post."
    else:
        response = "This job posting is probably an authentic post."
    
    
    bot.reply_to(message, response)

    

bot.infinity_polling()
    
