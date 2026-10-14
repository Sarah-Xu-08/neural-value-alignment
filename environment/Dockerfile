# hash:sha256:8488572e70a2da0466c66d192fa177de3b45d292701ae7c39d3a56f210ddb370
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==4.5.1 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    setuptools==75.1.0 \
    wheel==0.44.0
RUN pip3 install -U --no-cache-dir \
    addict==2.4.0 \
    click==8.1.7 \
    contourpy==1.3.0 \
    cycler==0.12.1 \
    fonttools==4.54.1 \
    fsspec==2024.9.0 \
    ipympl==0.9.7 \
    joblib==1.5.3 \
    jupyter==1.1.1 \
    jupyterlab==4.5.1 \
    kiwisolver==1.4.7 \
    lazy-loader==0.4 \
    matplotlib==3.7.5 \
    mkl-service==2.4.2 \
    mne==1.10.2 \
    numpy==1.26.4 \
    pandas==2.0.3 \
    pillow==12.0.0 \
    pingouin==0.5.5 \
    pooch==1.8.2 \
    protobuf==4.25.3 \
    pyparsing==3.2.0 \
    pyyaml==6.0.3 \
    requests==2.32.5 \
    scikit-image==0.21.0 \
    scikit-learn==1.8.0 \
    scipy==1.16.2 \
    seaborn==0.13.2 \
    statannotations==0.7.2 \
    statsmodels==0.14.1 \
    tensorboard==2.20.0 \
    threadpoolctl==3.6.0 \
    tomli==2.0.2 \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchcde==0.2.5 \
    torchdiffeq==0.2.4 \
    torchsde==0.2.6 \
    torchvision==0.19.1 \
    tqdm==4.66.5 \
    trampoline==0.1.2 \
    triton==3.0.0 \
    typeguard==2.13.3 \
    utilsd==0.0.18.post0 \
    yapf==0.40.2

COPY postInstall /
RUN /postInstall
