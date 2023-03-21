import telebot, os
from decouple import config
from my_code.models import predict

TELEGRAM_TOKEN = config('TELEGRAM_TOKEN')

bot = telebot.TeleBot(token=TELEGRAM_TOKEN)

FIELD_NAMES = [
    'title',
    'description'
]

queries = dict()
# this is a dictionary that will store for every chat the information of the currenty query
# title and description

def add_field(chat_id, field_name, content):
    if chat_id not in queries:
        queries[chat_id] = dict()
    queries[chat_id][field_name] = content


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    text = """
    Hi. This bot will accept the description of a Job Post and it will predict if it's a fraudulent or not.\n\
To use it, type:
    - /set_title TITLE to insert the title
    - /set_description DESCRIPTION to insert the description
    
    - /get_title to get the title of the current job post
    - /get_description to get the description of the current job post
    
    - /predict to predict whether it's fraudulent or not 
    """
    
    chat = message.chat
    
    for field in FIELD_NAMES:
        add_field(chat.id, field, 'missing')
    
    bot.reply_to(message, text)

@bot.message_handler(commands=['set_title'])
def handle_set_title(message):
    title = message.text
    title = title.replace('/set_title ', '')
    add_field(message.chat.id, 'title', title)
    response = """
    Title set correctly!\n\
You can still use:
    - /set_title TITLE to insert the title
    - /set_description DESCRIPTION to insert the description
    
    - /get_title to get the title of the current job post
    - /get_description to get the description of the current job post
    
    - /predict to predict whether it's fraudulent or not 
    """
    bot.reply_to(message, response)

@bot.message_handler(commands=['set_description'])
def handle_set_description(message):
    description = message.text
    description = description.replace('/set_description ', '')
    add_field(message.chat.id, 'description', description)
    response = """
    Description set correctly!\n\
You can still use:
    - /set_title TITLE to insert the title
    - /set_description DESCRIPTION to insert the description
    
    - /get_title to get the title of the current job post
    - /get_description to get the description of the current job post
    
    - /predict to predict whether it's fraudulent or not 
    """
    bot.reply_to(message, response)

@bot.message_handler(commands=['predict'])
def handle_prediction(message):
    prediction = predict(values=queries[message.chat.id])
    
    if prediction == 1:
        response = "This job posting is probably a fraudulent post."
    else:
        response = "This job posting is probably an authentic post."
    
    
    bot.send_message(message.chat.id, response)

@bot.message_handler(commands=['get_title'])
def handle_get_title(message):
    title = queries[message.chat.id]['title']
    bot.reply_to(message, title)
    
@bot.message_handler(commands=['get_description'])
def handle_get_description(message):
    description = queries[message.chat.id]['description']
    bot.reply_to(message, description)

bot.infinity_polling()
    
