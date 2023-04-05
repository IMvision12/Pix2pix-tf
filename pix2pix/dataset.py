import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

def decode_img(img):
    img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256,512))
    
    input_img, target_img = img[:,:256], img[:,256:]
    # Normalize
    input_img, target_img = normalize(input_img, target_img)
    # Convert to float32
    input_img = tf.cast(input_img, tf.float32)
    target_img = tf.cast(target_img, tf.float32)
    return target_img, input_img

def normalize(input_img, target_img):
    input_img = input_img / 127.5 - 1
    target_img = target_img / 127.5 - 1
    return input_img, target_img

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, 256, 256, 3])

    return cropped_image[0], cropped_image[1]

@tf.function()
def random_jitter(input_img, target_img):
    target_img, input_img = resize(input_img, target_img, 286, 286)
    target_img, input_img = random_crop(input_img, target_img)

    if tf.random.uniform(()) > 0.5:
        target_img = tf.image.flip_left_right(target_img)
        input_img = tf.image.flip_left_right(input_img)

    return input_img, target_img

def train_data(train_path):
    list_train = tf.data.Dataset.list_files(train_path, shuffle=True)
    train_ds = list_train.map(decode_img, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(random_jitter, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(1)
    train_ds = train_ds.prefetch(AUTOTUNE)
    return train_ds

def test_data(test_path):
    list_test = tf.data.Dataset.list_files(test_path, shuffle=True)
    test_ds = list_test.map(decode_img, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(1)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return test_ds