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
        
        #print img1.shape
        #print img2.shape
        concat_img_list.append(np.concatenate((img1,img2),axis=2))
    
    return concat_img_list

def image_loader(image_path, load_x, load_y, is_train = True):
    
    imgs = sorted(os.listdir(image_path))
    img_list = []
    for ele in imgs:
        img = Image.open(os.path.join(image_path, ele))
        # if is_train:
        #     img = img.resize((load_x, load_y), Image.BICUBIC)
        # else:
        #     img = img.resize((256,256), Image.BICUBIC)
        #img = img.resize((load_x, load_y), Image.BICUBIC)
        img_list.append(np.array(img))
    
    return img_list

def data_augument(lr_img, hr_img, aug):
    
    if aug < 4:
        lr_img = np.rot90(lr_img, aug)
        hr_img = np.rot90(hr_img, aug)
    
    elif aug == 4:
        lr_img = np.fliplr(lr_img)
        hr_img = np.fliplr(hr_img)
        
    elif aug == 5:
        lr_img = np.flipud(lr_img)
        hr_img = np.flipud(hr_img)
        
    elif aug == 6:
        lr_img = np.rot90(np.fliplr(lr_img))
        hr_img = np.rot90(np.fliplr(hr_img))
        
    elif aug == 7:
        lr_img = np.rot90(np.flipud(lr_img))
        hr_img = np.rot90(np.flipud(hr_img))
        
    return lr_img, hr_img

def batch_gen(blur_imgs, input_imgs, input_z, patch_size, batch_size, random_index, step, augment):
    
    img_index = random_index[step * batch_size : (step + 1) * batch_size]
    
    all_img_blur = []
    all_img_input = []
    all_img_z = []
    
    for _index in img_index:
        all_img_blur.append(blur_imgs[_index])
        all_img_input.append(input_imgs[_index])
        #all_img_focus.append(np.expand_dims(focus_imgs[_index],axis=3))
        all_img_z.append(input_z)
    
    blur_batch = []
    input_batch = []
    input_z_batch = []
    
    for i in range(len(all_img_blur)):
        
        ih, iw, _ = all_img_blur[i].shape
        ix = random.randrange(0, iw - patch_size +1)
        iy = random.randrange(0, ih - patch_size +1)
        
        # img_blur_in = all_img_blur[i][iy:iy + patch_size, ix:ix + patch_size]
        # img_sharp_in = all_img_sharp[i][iy:iy + patch_size, ix:ix + patch_size] 
        # img_focus_in = all_img_focus[i][iy:iy + patch_size, ix:ix + patch_size]
        img_blur_in = all_img_blur[i]
        img_input_in = all_img_input[i]
        #img_focus_in = all_img_focus[i]
        if augment:
            
            aug = random.randrange(0,8)
            img_blur_in, img_sharp_in = data_augument(img_blur_in, img_sharp_in, aug)

        blur_batch.append(img_blur_in)
        input_batch.append(img_input_in)
        #focus_batch.append(img_focus_in)
        input_z_batch.append(all_img_z)
        
    blur_batch = np.array(blur_batch)
    input_batch = np.array(input_batch)
    #focus_batch = np.array(focus_batch)
    input_z_batch = np.array(input_z_batch)
    #print("inp_z",input_z_batch.shape)
    
    return blur_batch, input_batch, input_z_batch

# def batch_gen(sharp_imgs, blur_imgs, focus_imgs, input_z, patch_size, batch_size, random_index, step, augment):
    
#     img_index = random_index[step * batch_size : (step + 1) * batch_size]
    
#     all_img_blur = []
#     all_img_sharp = []
#     all_img_focus = []
#     all_img_z = []
    
#     for _index in img_index:
#         all_img_blur.append(blur_imgs[_index])
#         all_img_sharp.append(sharp_imgs[_index])
#         all_img_focus.append(np.expand_dims(focus_imgs[_index],axis=3))
#         all_img_z.append(input_z)
    
#     blur_batch = []
#     sharp_batch = []
#     focus_batch = []
#     input_z_batch = []
    
#     for i in range(len(all_img_blur)):
        
#         ih, iw, _ = all_img_blur[i].shape
#         ix = random.randrange(0, iw - patch_size +1)
#         iy = random.randrange(0, ih - patch_size +1)
        
#         # img_blur_in = all_img_blur[i][iy:iy + patch_size, ix:ix + patch_size]
#         # img_sharp_in = all_img_sharp[i][iy:iy + patch_size, ix:ix + patch_size] 
#         # img_focus_in = all_img_focus[i][iy:iy + patch_size, ix:ix + patch_size]
#         img_blur_in = all_img_blur[i]
#         img_sharp_in = all_img_sharp[i]
#         img_focus_in = all_img_focus[i]
#         if augment:
            
#             aug = random.randrange(0,8)
#             img_blur_in, img_sharp_in = data_augument(img_blur_in, img_sharp_in, aug)

#         blur_batch.append(img_blur_in)
#         sharp_batch.append(img_sharp_in)
#         focus_batch.append(img_focus_in)
#         input_z_batch.append(all_img_z)
        
#     blur_batch = np.array(blur_batch)
#     sharp_batch = np.array(sharp_batch)
#     focus_batch = np.array(focus_batch)
#     input_z_batch = np.array(input_z_batch)
#     #print("inp_z",input_z_batch.shape)
    
#     return blur_batch, sharp_batch, focus_batch, input_z_batch



# In[ ]:


def recursive_forwarding(blur, chopSize, session, net_model, chopShave = 20):
    b, h, w, c = blur.shape
    wHalf = math.floor(w / 2)
    hHalf = math.floor(h / 2)
    
    wc = wHalf + chopShave
    hc = hHalf + chopShave
    
    inputPatch = np.array((blur[:, :hc, :wc, :], blur[:, :hc, (w-wc):, :], blur[:,(h-hc):,:wc,:], blur[:,(h-hc):,(w-wc):,:]))
    outputPatch = []
    if wc * hc < chopSize:
        for ele in inputPatch:
            output = session.run(net_model.output, feed_dict = {net_model.blur : ele})
            outputPatch.append(output)

    else:
        for ele in inputPatch:
            output = recursive_forwarding(ele, chopSize, session, net_model, chopShave)
            outputPatch.append(output)
    
    upper = np.concatenate((outputPatch[0][:,:hHalf,:wHalf,:], outputPatch[1][:,:hHalf,wc-w+wHalf:,:]), axis = 2)
    rower = np.concatenate((outputPatch[2][:,hc-h+hHalf:,:wHalf,:], outputPatch[3][:,hc-h+hHalf:,wc-w+wHalf:,:]), axis = 2)
    output = np.concatenate((upper,rower),axis = 1)
    
    return output

