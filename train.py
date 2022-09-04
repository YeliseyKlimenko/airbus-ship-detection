import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

# Importing the dataset
ships = pd.read_csv('train_ship_segmentations_v2.csv')

# Dropping the NaN values
ships_nonempty = ships.dropna().reset_index(drop=True)

# Rle decoder to process  and resize the masks


def rle_decode(mask_rle, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    img = Image.fromarray(img.reshape(shape).T)
    newimg = img.resize((128, 128))
    return newimg


# Constructing the Y training set
Y_train = np.zeros((42556, 128, 128), dtype=np.uint8)
Y_train[0] = rle_decode(ships_nonempty["EncodedPixels"][0], (768, 768))
j = 0
for i in np.arange(1, ships_nonempty.ImageId.size):
    if ships_nonempty.ImageId[i] == ships_nonempty.ImageId[i - 1]:
        mask1 = Y_train[j]
        mask2 = rle_decode(ships_nonempty["EncodedPixels"][i], (768, 768))
        Y_train[j] = (mask1 == 1) | (mask2 == 1)
    else:
        j += 1
        mask2 = rle_decode(ships_nonempty["EncodedPixels"][i], (768, 768))
        Y_train[j] = mask2

print('Y_train finished')

# Dropping duplicate images
ships_nonempty = ships_nonempty.drop_duplicates(subset='ImageId', keep='first').reset_index(drop=True)

# Constructing the X training set
X_train = np.zeros((len(ships_nonempty.ImageId), 128, 128, 3), dtype=np.uint8)
i = 0
for trainid in ships_nonempty.ImageId:
    img = Image.open(f'train_v2/{trainid}')
    newimg = img.resize((128, 128))
    X_train[i] = newimg
    i += 1

print('X_train finished')

# Saving 10k images (roughly 25%) for testing
Y_train = Y_train[:32556:]
X_train = X_train[:32556:]

# Input layer
inputs = tf.keras.layers.Input((128, 128, 3))

# Converting everything to float to avoid issues with keras
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

# Contraction path (encoder) for the u-net model
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path (decoder)
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

# Output layer
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Model compilation
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Setting up checkpoints, enabling TensorBoard, training the model
checkpointer = tf.keras.callbacks.ModelCheckpoint('airbuschallenge.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=100, epochs=4, callbacks=callbacks)

model.save_weights('weights')
