import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
import random


"""
    Training of the Network
    Parameters:
        args  - arguments defined in the main
        model - model defined in the build_graph  
"""
def train(args, model, sess, saver):
    
    if args.fine_tuning :
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s"%(args.pre_trained_model))
        
    num_imgs = len(os.listdir(args.train_Sharp_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)
    
    epoch = 0
    step = num_imgs // args.batch_size
    """
    input_imgs - Dimensions(BATCH_SIZE*256*256*4) --> concatenated RGB and FocusMeasure images along channel axis
    blur_imgs  - Dimensions(BATCH_SIZE*256*256*3) --> Label images
    input_z    - Magnification parameter used
    """
    input_imgs = util.image_loader_new(args.train_Sharp_path, args.train_focus_path, args.load_X, args.load_Y)
    blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y) #output
    input_z = 3
    
    while epoch < args.max_epoch:
        random_index = np.random.permutation(len(blur_imgs))
        s_time = time.time()
        for k in range(step):
            """
                utils.batch_gen function generates a batch of size args.batch_size that are fed to the network.
            """
            blur_batch, inp_batch, input_z_batch = util.batch_gen(blur_imgs, input_imgs, input_z, args.patch_size, args.batch_size, random_index, k)
            
            for t in range(args.critic_updates):
                _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict = {model.inp : inp_batch, model.blur : blur_batch, model.input_z: input_z_batch, model.epoch : epoch})
                
            _, G_loss = sess.run([model.G_train, model.G_loss], feed_dict = {model.inp : inp_batch, model.blur : blur_batch, model.input_z: input_z_batch, model.epoch : epoch})
                            
        e_time = time.time()
        
        if epoch % args.log_freq == 0:
            summary = sess.run(merged, feed_dict = {model.inp : inp_batch, model.blur : blur_batch, model.input_z: input_z_batch})
            train_writer.add_summary(summary, epoch)
            
            """
                Testing the model while training for knowing how the model is working after each iteration
            """
            if args.test_with_train:
                test(args, model, sess, saver, f, epoch, loading = False)

            print("%d training epoch completed" % epoch)
            print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
            print("Elpased time : %0.4f"%(e_time - s_time))

        """
            Saving the model every args.model_save_freq epochs into model1 directory
        """
        if ((epoch) % args.model_save_freq ==0):
            saver.save(sess, './model1/DeblurrGAN', global_step = epoch)
        
        epoch += 1

    """
        Saving the model obtained after the last iteration
    """
    saver.save(sess, './model1/DeblurrGAN_last')
        
"""
    Test function used to test the model while training
    Parameters:
        args  - arguments defined in the main
        model - model defined in the build_graph
        step  - epoch during training
"""        
def test(args, model, sess, saver, step = -1, loading = False):
        
    if loading:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for test!")
        print("model path is %s"%args.pre_trained_model)
        
    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))
    focus_img_name = sorted(os.listdir(args.test_focus_path))
    
    if args.in_memory :
        
    input_imgs = util.image_loader_new(args.test_Sharp_path, args.test_focus_path, args.load_X, args.load_Y)
    blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train = False)
    
    index = 0
    input_z = np.zeros(args.z_dim)
    input_z[0] = 3
    input_z = np.expand_dims(input_z, axis = 0)
    
    blur = np.expand_dims(blur_imgs[index], axis = 0)
    inp = np.expand_dims(input_imgs[index], axis = 0)
   
    output = sess.run(model.output, feed_dict = {model.inp : inp, model.blur : blur, model.input_z : input_z})
    if args.save_test_result:
        output = Image.fromarray(output[0])
        split_name = blur_img_name[index].split('.')
        output.save(os.path.join(args.result_path, 'epoch%d_%s_blur.png'%(step,''.join(map(str, split_name[:-1])))))


"""
    test_only function is used to test the model on test data after training
    Parameters:
        args  - arguments defined in the main
        model - model defined in the build_graph
    saver restores the saved model from args.pre_trained_model directory
"""            
def test_only(args, model, sess, saver):
    
    saver.restore(sess,args.pre_trained_model)
    graph = sess.graph
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    blur_img_name = sorted(os.listdir(args.test_Blur_path))

        
    input_imgs = util.image_loader_new(args.test_Sharp_path, args.test_focus_path, args.load_X, args.load_Y)
    input_z = np.zeros(args.z_dim)
    input_z[0] = 3
    input_z = np.expand_dims(input_z, axis = 0)
    for i, ele in enumerate(input_imgs):
        inp = np.expand_dims(ele, axis = 0)
        
        if args.chop_forward:
            output = util.recursive_forwarding(inp, args.chop_size, sess, model, args.chop_shave)
            output = Image.fromarray(output[0])
        
        else:
            output = sess.run(model.output, feed_dict = {model.inp : inp, model.input_z : input_z})
            output = Image.fromarray(output[0])
        
        split_name = blur_img_name[i].split('.')
        output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))


