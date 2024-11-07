#!/bin/bash

VERSION="v1.00"

docker build -t doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION . && \
docker push doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION && \

docker tag doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION doduo1.umcn.nl/nnunet_for_pathology/sol2:latest
docker push doduo1.umcn.nl/nnunet_for_pathology/sol2:latest
