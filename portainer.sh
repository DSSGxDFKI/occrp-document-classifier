#!/usr/bin/bash
systemctl --user enable --now podman.socket

docker run -d \
  -p 9000 \
  --security-opt label=disable \
  --name=portainer \
  --restart=always \
  -v /run/user/$(id -u)/podman/podman.sock:/var/run/docker.sock:Z \
  -v portainer_data:/data \
  portainer/portainer-ce:2.13.1

echo "Portainer is available in url http://$(docker port portainer 9000)"