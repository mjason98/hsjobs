import telebot, os
from decouple import config

TELEGRAM_TOKEN = config('TELEGRAM_TOKEN')

bot = telebot.TeleBot(token=TELEGRAM_TOKEN)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    text = """s
Hi. This bot will accept the description of a Job Post and it will predict if it's a fraudulent or not.
To use it, just send the description as a text file.
    """
    
    bot.reply_to(message, text)

@bot.message_handler(func= lambda message: message.content_type == "text")
def handle_job_post(message):
    bot.reply_to(message, message.text)

    

bot.infinity_polling()
    
