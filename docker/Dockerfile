FROM python:3.5

RUN apt update && apt upgrade -y
RUN apt install wget build-essential gcc -y

RUN pip install --upgrade pip
RUN pip install pytest coverage
RUN pip install numpy pandas scikit-learn joblib
# RUN pip install spherecluster
RUN git clone https://github.com/druogury/spherecluster.git
RUN cd spherecluster
RUN ls
RUN python setup.py install
