FROM python:3.13-alpine

WORKDIR /app

COPY app.py .
COPY score.py .
COPY best_model.pkl .
COPY vectorizer.pkl .
COPY requirements.txt .

RUN apk update
RUN apk add g++ gcc make

RUN pip install --root-user-action=ignore -r requirements.txt

CMD ["python", "app.py"]