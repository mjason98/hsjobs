FROM python:3.9-bullseye

WORKDIR /app
COPY ./requirements-bot.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "my_code.chatbot"]
