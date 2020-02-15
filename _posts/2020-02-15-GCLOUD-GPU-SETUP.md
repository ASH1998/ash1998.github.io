---
layout: post
title:  "GPU-Server-SetUp-for-DeepLearning"
date:   2020-02-15
desc: "Gcloud server setup"
keywords: "Ashutosh,Python,ML, Tech"
categories: [Deep Learning, Python, Machine Learning]
tags: [GPU, Python, Deep-Learning]
icon: icon-html
---

# Introduction 
This blog is about how to setup gpu drivers and nvidia-smi for Deep Learning in ubuntu servers.        
This guide will be using conda install of python as it is easier to maintain and keep track of correct versions using it.

## Dependencies :
No particular dependencies except this has been only tested on ubuntu 16.04 LTS and 18.04 LTS.        
(Should work in other versions as well)

## Steps : 
- Download conda :        
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

- Install the latest version of conda :       
`bash Miniconda3-latest-Linux-x86_64.sh`            
Here during installing add the path to ~/bashrc, so that you can call conda using `bash`.

- Now create a new conda virtual environment for TF-GPU       
`conda create -n TFGPU -c defaults tensorflow-gpu`   
(this will install the tensorflow base gpu, cudnn and all required modules)

- Then after this start it by using    
`conda activate TFGPU`       
(But at this point there are no nvidia drivers so module wont use gpu)

- Now install nvidia-drivers ( this takes a little bit time)

```
#!/bin/bash                
	echo "Checking for CUDA and installing."          
	# Check for CUDA and try to install.             
	if ! dpkg-query -W cuda; then
	  # The 16.04 installer works with 16.10.
	  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	  apt-get update
	  # apt-get install cuda -y
	  sudo apt-get install cuda-8-0
fi
```

- Then after it is complete you can check the status by   
`nvidia-smi` :    
Result should be something like 
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   64C    P0    28W /  70W |    173MiB / 15109MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1945      G   /usr/lib/xorg/Xorg                            62MiB |
|    0      4183      C   python                                        99MiB |
+-----------------------------------------------------------------------------+