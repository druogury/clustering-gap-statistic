FROM python:3.5

RUN apt update && apt upgrade -y
RUN apt install wget build-essential gcc -y

RUN pip install --upgrade pip
RUN pip install pytest coverage
RUN pip install numpy pandas scikit-learn joblib scipy
# RUN pip install spherecluster
RUN git clone https://github.com/druogury/spherecluster.git
RUN ls spherecluster
RUN cd spherecluster && python setup.py install
