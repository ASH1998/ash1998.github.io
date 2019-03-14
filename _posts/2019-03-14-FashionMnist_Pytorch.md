---
layout: post
title:  "Pytorch"
date:   2018-10-18
desc: "CNN on Fashion Mnist using Pytorch"
keywords: "Pytorch,Fashion"
categories: [tech]
tags: [Ashutosh, Life, resume, Pytorch, Mnist]
icon: icon-html
---
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import gzip\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import argparse\n",
    "\n",
    "import tqdm\n",
    "\n",
    "torch.set_printoptions(linewidth=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_mnist(path, kind='train'):\n",
    "\n",
    "    \"\"\"Load MNIST data from `path`\"\"\"\n",
    "    labels_path = os.path.join(path,\n",
    "                               '%s-labels-idx1-ubyte.gz'\n",
    "                               % kind)\n",
    "    images_path = os.path.join(path,\n",
    "                               '%s-images-idx3-ubyte.gz'\n",
    "                               % kind)\n",
    "\n",
    "    with gzip.open(labels_path, 'rb') as lbpath:\n",
    "        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,\n",
    "                               offset=8)\n",
    "\n",
    "    with gzip.open(images_path, 'rb') as imgpath:\n",
    "        images = np.frombuffer(imgpath.read(), dtype=np.uint8,\n",
    "                               offset=16).reshape(len(labels), 784)\n",
    "\n",
    "    return images, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images, train_labels = load_mnist(r\"E:\\DATA SETS & compressed\\fashion data\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((60000, 784), (60000,))"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_images.shape, train_labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_images, test_labels = load_mnist(r'E:\\DATA SETS & compressed\\fashion data', kind='t10k')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((10000, 784), (10000,))"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_images.shape, test_labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images = train_images.reshape((60000, 28, 28))\n",
    "test_images=test_images.reshape((10000, 28, 28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = train_images[4545]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(a)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = []\n",
    "test_data = []\n",
    "for i in range(len(train_images)):\n",
    "    train_data.append([train_images[i], train_labels[i]])\n",
    "for i in range(len(test_images)):\n",
    "    test_data.append([test_images[i], test_labels[i]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)\n",
    "testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=100, )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "i1, l1 = next(iter(trainloader))\n",
    "i2, l2 = next(iter(testloader))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor(5, dtype=torch.uint8)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAADYxJREFUeJzt3VuMXeV5xvHnYWY8xmMMtogdc2gN1AmgqHWqqSF1VLlCpARFMqiiiishR0JxlAaJSFwUcRNuKqGqQKoqSuUUC1dKSKMSgi9oi2VR0bSJxYBQMHGCHccBH7ChTmNjgj2HtxezHE3M7G9v79Pa1vv/Sdbee71rzXq9x4/34VtrfY4IAcjnorobAFAPwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+IKnhfu5sgUdjocb6uUsglfd1SmfitFtZt6Pw275N0t9LGpL0TxHxcGn9hRrTTb6lk10CKNgVO1tet+23/baHJH1N0qcl3Shpo+0b2/15APqrk8/8ayXti4j9EXFG0rclbehOWwB6rZPwXynpzTmPD1bLfovtzbYnbE9M6nQHuwPQTZ2Ef74vFT5wfnBEbImI8YgYH9FoB7sD0E2dhP+gpKvnPL5K0uHO2gHQL52E/0VJq21fY3uBpM9K2t6dtgD0WttDfRExZfteSf+h2aG+rRHxWtc6A9BTHY3zR8Szkp7tUi8A+ojDe4GkCD+QFOEHkiL8QFKEH0iK8ANJEX4gKcIPJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iqo1l6bR+QdFLStKSpiBjvRlMAeq+j8Ff+NCLe6cLPAdBHvO0Hkuo0/CHpOdsv2d7cjYYA9Eenb/vXRcRh28sl7bD9k4h4Ye4K1X8KmyVpoRZ1uDsA3dLRK39EHK5uj0l6WtLaedbZEhHjETE+otFOdgegi9oOv+0x25ecvS/pU5J2d6sxAL3Vydv+FZKetn3253wrIv69K10B6Lm2wx8R+yX9QRd7AdBHDPUBSRF+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iK8ANJEX4gKcIPJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+ICnCDyRF+IGkCD+QFOEHkmoafttbbR+zvXvOsmW2d9jeW90u7W2bALqtlVf+JyTdds6yByTtjIjVknZWjwFcQJqGPyJekHT8nMUbJG2r7m+TdEeX+wLQY+1+5l8REUckqbpd3r2WAPTDcK93YHuzpM2StFCLer07AC1q95X/qO2VklTdHmu0YkRsiYjxiBgf0WibuwPQbe2Gf7ukTdX9TZKe6U47APqllaG+JyX9QNJHbR+0fY+khyXdanuvpFurxwAuIE0/80fExgalW7rcy0DzaOEjy/R0T/cdU1M9+9n/d/cnivXVX9xTrO954oZifckbjXtf+Nap4rZHP3FpsT61yMX6Ff/5q4a1n9+5pLht/F65t6l3Li7Wh5adLtYvW/Jew9qyz7xe3LZbOMIPSIrwA0kRfiApwg8kRfiBpAg/kFTPD+8dFB5u8lcdGiqW43R56GZQnfmz8WL94ruPFOs/2HV9sT50bXn/n7vvuYa12xe/Vtz20PTiYn3MZ4r1F++5pmHtxoWHitv+6/E/KtZHVpeHd58/tLpYn3zu8kK1yVCfC0OcUd50Ll75gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApR5zHwGCHlnhZ3OQL80zgmU+uaVh764/LlydzkzN+J8fK9fc/XD6l11ONx30v2V8+fuHDX/2f8s5rtPS/lxXrf7n8h8X611Z/pO19D91QHqef3rO37Z/dS7tip07E8fK5zhVe+YGkCD+QFOEHkiL8QFKEH0iK8ANJEX4gqf6ez+8WzqsviNIlsjs8XuFnj9xcrI+uOtmwtuq+XxS3PfX7VxTrFx96t1jXvjeK5ZlT5ctM91Tp3HKpo9/LxA/L4/SP3bW9yU9of5y/03H84qXeOxRnCtcx4Hx+AM0QfiApwg8kRfiBpAg/kBThB5Ii/EBSTQfdbW+V9BlJxyLiY9WyhyR9XtLb1WoPRsSzTfcWvZ1uuhP7Nv5jsX77+j9vWJs6WL4G/GiT+kyxOuB6eD2I6+4vn6//0zvK02yXr2/f2+tYXAjzPLTyyv+EpNvmWf5YRKyp/jQPPoCB0jT8EfGCpON96AVAH3Xymf9e2z+yvdX20q51BKAv2g3/1yVdJ2mNpCOSHmm0ou3NtidsT0xq8D8HAVm0Ff6IOBoR0xExI+kbktYW1t0SEeMRMT6i3p3sAOD8tBV+2yvnPLxT0u7utAOgX1oZ6ntS0npJl9s+KOkrktbbXqPZEwgPSPpCD3sE0ANNwx8RG+dZ/Hg7O/PoAg1ftaphfXLFpcXth98+0bA2ve/nxW1L192XpH/45S+L9enXf9a42OE57U2vcTBUvvb+RYVzxz1WnlMgLrukWJ9ZtKBYn15crk+ONf67TS8sv/Fc+L+F89YlnZgpz2N/6t+uaVi79K/KR1ecuv5DxfroO+8X60Mny3VNF/Y/XP59x/7G13fw+y1dsl8SR/gBaRF+ICnCDyRF+IGkCD+QFOEHkurrFN2XLroibr7+8w3rb627rLj9rz7SeHjkog7PFJ4ZKT8PMdb4suELFpeHpIaHy3N0T02Vh3YmT5eHAocOt3/kZLPnbaY8kqehX5eHloZ/3bgWTUalmvV2emn5d3btzY2HxBYNl39ni4Yni/X3pkaa1JsMkUbj192Ri8r/Xo48taphbe+/PKr3jr7JFN0AGiP8QFKEH0iK8ANJEX4gKcIPJEX4gaT6Os6/xMviJt/St/0B2eyKnToRxxnnB9AY4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+ICnCDyTVNPy2r7b9vO09tl+zfV+1fJntHbb3VrdLe98ugG5p5ZV/StL9EXGDpJslfcn2jZIekLQzIlZL2lk9BnCBaBr+iDgSES9X909K2iPpSkkbJG2rVtsm6Y5eNQmg+87rM7/tVZI+LmmXpBURcUSa/Q9C0vJuNwegd1oOv+3Fkp6S9OWIOHEe2222PWF7YlKn2+kRQA+0FH7bI5oN/jcj4rvV4qO2V1b1lZKOzbdtRGyJiPGIGB9R+xNKAuiuVr7tt6THJe2JiEfnlLZL2lTd3yTpme63B6BXynM/z1on6W5Jr9p+pVr2oKSHJX3H9j2S3pB0V29aBNALTcMfEd+X1Og64FyEH7hAcYQfkBThB5Ii/EBShB9IivADSRF+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iK8ANJEX4gKcIPJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+IKmm4bd9te3nbe+x/Zrt+6rlD9k+ZPuV6s/tvW8XQLcMt7DOlKT7I+Jl25dIesn2jqr2WET8Xe/aA9ArTcMfEUckHanun7S9R9KVvW4MQG+d12d+26skfVzSrmrRvbZ/ZHur7aUNttlse8L2xKROd9QsgO5pOfy2F0t6StKXI+KEpK9Luk7SGs2+M3hkvu0iYktEjEfE+IhGu9AygG5oKfy2RzQb/G9GxHclKSKORsR0RMxI+oaktb1rE0C3tfJtvyU9LmlPRDw6Z/nKOavdKWl399sD0CutfNu/TtLdkl61/Uq17EFJG22vkRSSDkj6Qk86BNATrXzb/31Jnqf0bPfbAdAvHOEHJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IyhHRv53Zb0v6xZxFl0t6p28NnJ9B7W1Q+5LorV3d7O13I+JDrazY1/B/YOf2RESM19ZAwaD2Nqh9SfTWrrp6420/kBThB5KqO/xbat5/yaD2Nqh9SfTWrlp6q/UzP4D61P3KD6AmtYTf9m22f2p7n+0H6uihEdsHbL9azTw8UXMvW20fs717zrJltnfY3lvdzjtNWk29DcTMzYWZpWt97gZtxuu+v+23PSTpdUm3Sjoo6UVJGyPix31tpAHbBySNR0TtY8K2/0TSu5L+OSI+Vi37W0nHI+Lh6j/OpRHx1wPS20OS3q175uZqQpmVc2eWlnSHpM+pxueu0NdfqIbnrY5X/rWS9kXE/og4I+nbkjbU0MfAi4gXJB0/Z/EGSduq+9s0+4+n7xr0NhAi4khEvFzdPynp7MzStT53hb5qUUf4r5T05pzHBzVYU36HpOdsv2R7c93NzGNFNW362enTl9fcz7maztzcT+fMLD0wz107M153Wx3hn2/2n0EaclgXEX8o6dOSvlS9vUVrWpq5uV/mmVl6ILQ743W31RH+g5KunvP4KkmHa+hjXhFxuLo9JulpDd7sw0fPTpJa3R6ruZ/fGKSZm+ebWVoD8NwN0ozXdYT/RUmrbV9je4Gkz0raXkMfH2B7rPoiRrbHJH1Kgzf78HZJm6r7myQ9U2Mvv2VQZm5uNLO0an7uBm3G61oO8qmGMr4qaUjS1oj4m743MQ/b12r21V6ancT0W3X2ZvtJSes1e9bXUUlfkfQ9Sd+R9DuS3pB0V0T0/Yu3Br2t1+xb19/M3Hz2M3afe/ukpP+S9KqkmWrxg5r9fF3bc1foa6NqeN44wg9IiiP8gKQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8k9f9NfNndUncUFQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "num = 55\n",
    "plt.imshow(i1[num])\n",
    "print(l1[num])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n0 - tshirt\\n1 - trouser\\n2 - pullover\\n3 - dress\\n4 - coat\\n5 - sandal\\n6 - shirt\\n7 - sneakers\\n8 - bag\\n9 - ankle boot\\n'"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "0 - tshirt\n",
    "1 - trouser\n",
    "2 - pullover\n",
    "3 - dress\n",
    "4 - coat\n",
    "5 - sandal\n",
    "6 - shirt\n",
    "7 - sneakers\n",
    "8 - bag\n",
    "9 - ankle boot\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([5, 9, 3, 1, 5, 7, 5, 5, 9, 7, 4, 9, 9, 4, 8, 4, 0, 6, 4, 1, 9, 4, 7, 3, 0, 5, 6, 8, 3, 2, 5,\n",
       "        8, 7, 9, 5, 3, 9, 8, 5, 6, 3, 5, 4, 3, 2, 3, 3, 3, 8, 4, 2, 6, 1, 6, 6, 5, 1, 0, 0, 7, 7, 6,\n",
       "        4, 2, 1, 7, 7, 8, 5, 3, 4, 7, 2, 6, 2, 8, 2, 3, 1, 1, 3, 9, 6, 2, 1, 4, 1, 1, 3, 9, 0, 7, 0,\n",
       "        0, 4, 6, 7, 8, 7, 9], dtype=torch.uint8)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([100, 28, 28])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "i1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#start the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
