FROM ubuntu:16.04

RUN apt update && apt upgrade -y
RUN apt install wget build-essential gcc -y

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -O miniconda.sh --no-check-certificate
RUN bash miniconda.sh -bf -p /anaconda
RUN rm miniconda.sh
ENV PATH /anaconda/bin:$PATH
RUN pip install --upgrade pip
RUN pip install pytest coverage
RUN conda install numpy pandas scikit-learn joblib -y

# RUST
RUN wget https://static.rust-lang.org/dist/rust-nightly-x86_64-unknown-linux-gnu.tar.gz -O rust.tar.gz \
    && mkdir /opt/rust \
    && tar -xvzf rust.tar.gz -C /opt/rust
RUN bash /opt/rust/rust-nightly-x86_64-unknown-linux-gnu/install.sh
ENV PATH /usr/local/lib/rustlib/bin:$PATH
RUN pip install setuptools_rust

WORKDIR /workdir
COPY . /workdir
RUN bash -c "python setup.py build"
RUN python -c "from gap_statistic import OptimalK;print('imported without problem')"