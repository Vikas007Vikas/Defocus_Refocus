import tensorflow as tf

def fm_loss(blurIm, fmIm, gensharpIm):
    lossVal = tf.reduce_mean(tf.multiply(fmIm+1,tf.abs(blurIm-gensharpIm)))
    return lossVal
