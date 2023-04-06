#!/usr/bin/bash

# this env variable is used to change the dockerbuild output directory
export TMPDIR=/data/dssg/occrp/data/docker
podman run -d \
    --security-opt=label=disable \
    -v /data/dssg/occrp/data:/data \
    -v /data/dssg/occrp/data/:/data/dssg/occrp/data/ \
    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --cap-add SYS_ADMIN \
    $(podman build -f gpu.Dockerfile -t train-gpu -q) "$@"
