import tensorflow as tf

"""
    FOCUS_MEASURE LOSS
    Parameters:
        blurIm     - Label image
        gensharpIm - image generated from generator
        fmIm       - Focus measure
"""
def fm_loss(blurIm, fmIm, gensharpIm):
    lossVal = tf.reduce_mean(tf.multiply(fmIm+1,tf.abs(blurIm-gensharpIm)))
    return lossVal
