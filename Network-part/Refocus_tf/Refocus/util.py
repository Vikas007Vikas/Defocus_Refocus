from PIL import Image
import numpy as np
import random
import math
import os
import tensorflow as tf

def image_loader_new(image_path1, image_path2, load_x, load_y):
    imgs1 = sorted(os.listdir(image_path1))
    imgs2 = sorted(os.listdir(image_path2))
    concat_img_list = []
    for i in range(len(imgs1)):
        img1 = Image.open(os.path.join(image_path1,imgs1[i]))
        img2 = Image.open(os.path.join(image_path2,imgs2[i]))
        img1 = np.array(img1)
        #should verify this again for radiance
        #img2 = np.expand_dims(np.array(img2),axis=3)
        img2 = np.array(img2)
        
        concat_img_list.append(np.concatenate((img1,img2),axis=2))
    
    return concat_img_list

def image_loader(image_path, load_x, load_y, is_train = True):
    
    imgs = sorted(os.listdir(image_path))
    img_list = []
    for ele in imgs:
        img = Image.open(os.path.join(image_path, ele))
        img_list.append(np.array(img))
    
    return img_list

def batch_gen(blur_imgs, input_imgs, input_z, patch_size, batch_size, random_index, step):
    
    img_index = random_index[step * batch_size : (step + 1) * batch_size]
    
    all_img_blur = []
    all_img_input = []
    all_img_z = []
    
    for _index in img_index:
        all_img_blur.append(blur_imgs[_index])
        all_img_input.append(input_imgs[_index])
        all_img_z.append(input_z)
    
    blur_batch = []
    input_batch = []
    input_z_batch = []
    
    for i in range(len(all_img_blur)):
        
        ih, iw, _ = all_img_blur[i].shape
        ix = random.randrange(0, iw - patch_size +1)
        iy = random.randrange(0, ih - patch_size +1)
        
        img_blur_in = all_img_blur[i]
        img_input_in = all_img_input[i]
        
        blur_batch.append(img_blur_in)
        input_batch.append(img_input_in)
        input_z_batch.append(all_img_z)
        
    blur_batch = np.array(blur_batch)
    input_batch = np.array(input_batch)
    input_z_batch = np.array(input_z_batch)
    
    return blur_batch, input_batch, input_z_batch