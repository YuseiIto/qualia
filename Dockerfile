FROM python:3

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements.lock

CMD [ "python", "./bot.py" ]
