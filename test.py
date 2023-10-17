


from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
#from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json
import pickle
from keras.preprocessing.image import ImageDataGenerator
import keras



train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)
train_generator = train_datagen.flow_from_directory('CropDiseaseDataset/train', batch_size = 20, class_mode = 'categorical', target_size = (80, 80))
validation_generator = test_datagen.flow_from_directory('CropDiseaseDataset/train', batch_size = 20, class_mode = 'categorical', target_size = (80, 80))
base_model = keras.applications.InceptionResNetV2(input_shape = (80, 80, 3), include_top = False, weights = 'imagenet')
base_model.trainable = False
print(train_generator.class_indices)
add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(15, activation='softmax'))
model = add_model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
hist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
model.save_weights('model/inception_model_weights.h5')
model_json = model.to_json()
with open("model/inception_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('model/inception_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
