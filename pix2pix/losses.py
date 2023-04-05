import tensorflow as tf

loss_fn = tf.keras.losses.BinaryCrossentropy()
LAMBDA = 100
#Generator Losses
def l1_loss(target_img, generated_img):
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = mae(target_img, generated_img)
    return loss*LAMBDA

def gen_loss_from_disc(disc_generated):
    loss = loss_fn(tf.ones_like(disc_generated), disc_generated)
    return loss

#Discrimantor Losses
def Discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_fn(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_fn(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss