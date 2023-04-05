import tensorflow as tf
from tensorflow.keras import optimizers
from model import Discriminator, generator
from losses import l1_loss, gen_loss_from_disc, Discriminator_loss

generator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)

Generator = generator((256,256,3))
Discriminator = Discriminator((256,256,3))

@tf.function
def train_step(input_img, target_img):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #Generating images using input_images(fake)
        generated = Generator(input_img, training=True)
        
        disc_real = Discriminator([input_img, target_img], training=True)
        disc_generated = Discriminator([input_img, generated], training=True)
        
        #Generator Loss
        l1_loss_v = l1_loss(target_img, generated)
        gen_loss_disc = gen_loss_from_disc(disc_generated)
        Gen_loss = l1_loss_v + gen_loss_disc
        
        #Discriminator Loss
        Disc_loss = Discriminator_loss(disc_real, disc_generated)
        
    generator_gradients = gen_tape.gradient(Gen_loss,
                                            Generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          Generator.trainable_variables))
    
    discriminator_gradients = disc_tape.gradient(Disc_loss,
                                               Discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              Discriminator.trainable_variables))
    
    return Gen_loss, Disc_loss

def train_model(epochs, steps, dataset):
    for epoch in range(epochs):
        for sat_img, map_img in dataset.take(steps):
            gen_loss, disc_loss = train_step(sat_img, map_img)
        print(f"Epochs:{epoch} || Generator_Loss: {gen_loss:.3f} || Discriminator_Loss: {disc_loss:.3f}")
        
    Generator.save('/kaggle/working/generator',save_format='tf')
    Generator.save("/kaggle/working/generator.h5")

    Discriminator.save('/kaggle/working/discriminator/',save_format='tf')
    Discriminator.save("/kaggle/working/discriminator.h5")