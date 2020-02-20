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
Result should be something like this 
------------------------------------------------



![from https://github.com/torch/cutorch/issues/478 ](https://camo.githubusercontent.com/98d45361d9e667750865d2b2574d22921b875214/68747470733a2f2f7331342e706f7374696d672e696f2f36686d7a656f616f782f696d6167652e706e67)
