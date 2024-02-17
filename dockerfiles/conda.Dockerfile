# syntax=docker/dockerfile:1.5.0
#FROM docker.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM docker.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

#RUN echo "trying to fix nvidia stuff" && \
#    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean 

RUN <<EOF
#!/bin/bash
apt update -q 
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget 
apt clean 
rm -rf /var/lib/apt/lists/*
EOF

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Choose conda version based on arguments
#.. [CondaHashes] https://docs.conda.io/projects/miniconda/en/latest/miniconda-hashes.html
#.. [CondaInstallers] https://docs.conda.io/en/latest/miniconda.html#linux-installers
ARG CONDA_PY_VERSION=py311
ARG CONDA_VERSION=23.10.0-1
ARG CONDA_SHA256=d0643508fa49105552c94a523529f4474f91730d3e0d1f168f1700c43ae67595

RUN <<EOF
#!/bin/bash

OS=Linux
ARCH=x86_64
CONDA_KEY="Miniconda3-${CONDA_PY_VERSION}_${CONDA_VERSION}-${OS}-${ARCH}"
CONDA_INSTALL_SCRIPT_FNAME="${CONDA_KEY}.sh"
CONDA_URL="https://repo.anaconda.com/miniconda/${CONDA_INSTALL_SCRIPT_FNAME}"

# Download conda and verify its hash
wget --quiet "${CONDA_URL}" -O miniconda.sh 
echo "${CONDA_SHA256}  miniconda.sh" > miniconda.sha256
if ! sha256sum --status -c miniconda.sha256; then
    echo "HASH DOES NOT MATCH!"
    exit 22
fi 

# Install conda into /opt
mkdir -p /opt 
sh miniconda.sh -b -p /opt/conda 
rm miniconda.sh miniconda.sha256 

ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
echo "conda activate watch" >> ~/.bashrc 
find /opt/conda/ -follow -type f -name '*.a' -delete
find /opt/conda/ -follow -type f -name '*.js.map' -delete 

/opt/conda/bin/conda clean -afy
EOF


################
### __DOCS__ ###
################
RUN <<EOF
echo '
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/

# To Build:

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t conda_base_image \
    -f ./dockerfiles/conda.Dockerfile .


# To Test
docker run --runtime=nvidia -it conda_base_image bash

'
EOF
