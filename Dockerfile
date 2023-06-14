# main stage
FROM --platform=$TARGETPLATFORM python:3.6.9-slim AS main

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,source=/wheel,target=/mnt/pypi \
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends zip unzip htop screen libgl1-mesa-glx libgl1 libglib2.0-0 libegl1

# python packages
RUN pip3 install --upgrade pip
RUN --mount=type=bind,source=/wheel,target=/mnt/pypi \
    --mount=type=bind,source=/requirements.txt,target=/mnt/requirements.txt \
    pip3 install --no-cache-dir -r /mnt/requirements.txt -f /mnt/pypi/
RUN --mount=type=bind,source=/wheel,target=/mnt/pypi \
    --mount=type=bind,source=/requirements_offline.txt,target=/mnt/requirements_offline.txt \
    pip3 install --no-cache-dir -r /mnt/requirements_offline.txt -f /mnt/pypi/ --no-index
RUN echo "/usr/lib/python3.6/dist-packages" > /usr/local/lib/python3.6/site-packages/tensorrt.pth

# app
ARG TARGETARCH
ADD app.tar /app
ADD yolov7_weight.tar /app
# COPY weight/$TARGETARCH /app/weight/device/

# cuda
RUN echo "/usr/lib/`arch`-linux-gnu/tegra" >> /etc/ld.so.conf.d/nvidia-tegra.conf && \
    echo "/usr/local/cuda-10.2/targets/`arch`-linux/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda-11.1/targets/`arch`-linux/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig

WORKDIR /app
ENV PYTHONPATH="/app/"
ENV LC_ALL="C.UTF-8" LANG="C.UTF-8"
# ENV FLASK_APP="website.Flask_Server_all"
# ENV FLASK_RUN_PORT=5020
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["sh", "-c", "gunicorn svc:app --workers ${GUNICORN_W:-1} --threads 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-5109}"]
# gunicorn svc:app --workers 4 --threads 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5109