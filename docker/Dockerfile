FROM python:3.5

RUN apt update && apt upgrade -y
RUN apt install wget build-essential gcc -y

RUN pip install --upgrade pip
RUN pip install pytest coverage numpy pandas scikit-learn joblib spherecluster
