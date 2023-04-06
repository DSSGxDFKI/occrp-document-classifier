#!/usr/bin/bash

# this env variable is used to change the dockerbuild output directory
export TMPDIR=/data/dssg/occrp/data/docker
name1="EfficientNetB0"
name2="EfficientNetB0BW"
name3="EfficientNetB4" 
name4="EfficientNetB4BW"

podman run -d \
    --security-opt=label=disable \
    -v /data/dssg/occrp/data:/data \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --cap-add SYS_ADMIN \
    --name  $name1 \
    $(podman build -f gpu.Dockerfile -t train-gpu -q) train-feature-extraction --model-name EfficientNetB4

podman run -d \
    --security-opt=label=disable \
    -v /data/dssg/occrp/data:/data \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --cap-add SYS_ADMIN \
    --name $name2 \
    $(podman build -f train_gpu.Dockerfile -t train-gpu -q) train-feature-extraction --model-name EfficientNetB0BW
podman wait --name=$name2 \

podman run -d \
    --security-opt=label=disable \
    -v /data/dssg/occrp/data:/data \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --cap-add SYS_ADMIN \
    --name  $name3 \
    $(podman build -f train_gpu.Dockerfile -t train-gpu -q) train-feature-extraction --model-name EfficientNetB4
podman wait --name=$name3 \

podman run -d \
    --security-opt=label=disable \
    -v /data/dssg/occrp/data:/data \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --cap-add SYS_ADMIN \
    --name $name4 \
    $(podman build -f gpu.Dockerfile -t train-gpu -q) train-feature-extraction --model-name EfficientNetB4BW

podman wait $name4

