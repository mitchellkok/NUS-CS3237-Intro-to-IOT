from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers iomport Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDateGenerator
from tensorflow.keras.optimizers import SGD
import os.path

MODEL_FILE = "flowers.hd5"

# create a model if none exists. Freese all training except in
# newly attached output layers. We can specify the number of nodes 
# in the hidden penultimate layer, and the number of output
# categories.

def create_model(num_hidden, num_classes):
    base_model = InceptionV3(include_top = False, weights = 'imagenet')
    
    # add new layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # freeze layers in base model
    for layer in base_model.layers:
        layer.trainable = False

    # create new  model
    model = Model(inputs = base_model.input, outputs = predictions)

    return model


# load an existing model file, then sets only the last
# 3 layers (which we added) to trainable
def load_existing(model_file):
    # load the model
    model = load_model(model_file)

    # set only last 3 layers as trainable
    numlayers = len(model.layers)

    for layer in model.layers[:numlayers-3]:
        layer.trainable = False

    for layers in model.layers[numlayers-3:]:
        layer.trainable = True

    return model

def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20):
    # check if model exists, then create or load model
    if os.path.exists(model.file):
        print("\n*** Existing model found at %s. Loading. ***\n\n" % model_file)
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model(num_hidden, num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # create a checkpoint
    checkpoint = ModelCheckpoint(model_file)

    # object to produce additional images
    train_datagen = ImageDataGenerator(\
        rescale=1./255,\
        shear_range=0.2,\
        zoom_range=0.2,\
        horizontal_flip = True)

    #scale data to [0,1]
    test_datagen = ImageDataGenerator(rescale=1./255)

    # create ImageDataGenerator for training data
    train_generator = train_datagen,flow_from_directory(\
        train_path,\
        target_size=(249,249),
        batch_size=32,
        class_mode = "categorical")

    # create ImageDataGenerator for testing data
    validation_generator = test_datagen.flow_from_directory(\
        validation_path,\
        target_size=(249,249),
        batch_size=32,
        class_mode = 'categorical')
        
    # train the network
    model.fit(\
        train_generator,\
        steps_per_epoch = steps,\
        epochs = num_epochs,\
        callbacks = [checkpoint],\
        validation_data = validation_generator,\
        validation_steps = 50)

    # train last two layers
    for layer in model.layers[:249]:
        layer.trainable = False

    for layer in model.layers[249:]:
        layer.trainable = True

    # compile model
    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy')

def main():
    train(MODEL_FILE, train_path="flower_photos", validation_path="flower_photos", steps=120, num_epochs=10)

if __name__ == "__main__":
    main()