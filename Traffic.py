import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras.layers import Conv2D,Input,Dense,MaxPool2D,BatchNormalization,GlobalAvgPool2D,Flatten
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

#a function to split images of each class into train and val sets
def split_data(path_to_data,path_to_save_train,path_to_save_val,split_size=0.1):
    #get the folders within the train data folder
    folders=os.listdir(path_to_data)
    for folder in folders:
        full_path=os.path.join(path_to_data,folder)
        #get the list of images within each folder
        images_paths=glob.glob(os.path.join(full_path,"*.png"))

        #split the data set into train and val
        x_train,x_val=train_test_split(images_paths,test_size=split_size,random_state=0)
        #for train set
        for x in x_train:
            #get the name of the image
            path_to_folder=os.path.join(path_to_save_train,folder)

            #create the folder if it doesn't exist
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)

            shutil.copy(x,path_to_folder)
        #for val set
        for x in x_val:
            #get the name of the image
            path_to_folder=os.path.join(path_to_save_val,folder)

            #create the folder if it doesn't exist
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)

            shutil.copy(x,path_to_folder)

            
#order the test set like the train set
def order_test_set(path_to_image,path_to_csv):
    testset={}     
    try:
        with open(path_to_csv,'r') as csvfile:
            reader=csv.reader(csvfile,delimiter=',')
            for i,row in enumerate(reader):
                if i==0:
                     continue
                img_name=row[-1].replace('Test/','')
                label=row[-2]
                path_to_folder=os.path.join(path_to_image,label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)
                img_full_path=os.path.join(path_to_image,img_name)
                shutil.move(img_full_path,path_to_folder)
    except:
        print("[INFO]:Error in reading the csv file")
   

#define the model
def streetsigns_model(num_classes):
    inputlayer=Input(shape=(60,60,3))

    x=Conv2D(32,(3,3),activation='relu')(inputlayer)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Conv2D(64,(3,3),activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Conv2D(128,(3,3),activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=GlobalAvgPool2D()(x)
    x=Dense(128,activation='relu')(x)
    x=Dense(num_classes,activation='softmax')(x)

    return Model(inputs=inputlayer,outputs=x)

#define the function to generate the data
def create_generator(batch_size,train_data_path,val_data_path,test_data_path):
    preprocessor=ImageDataGenerator(
        rescale=1/255,
    )
    train_generator=preprocessor.flow_from_directory(
        train_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    val_generator=preprocessor.flow_from_directory(
        val_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    test_generator=preprocessor.flow_from_directory(
        test_data_path,
        class_mode='categorical',
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    return train_generator,val_generator,test_generator

    

split_data("Train","training_data/train","training_data/val",0.1)
order_test_set("Test","Test.csv")


#initialise the data generators
train_gen,val_gen,test_gen=create_generator(64,'training_data/train','training_data/val','Test')
optimizer =  tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
model=streetsigns_model(train_gen.num_classes)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen,epochs=15,batch_size=64,validation_data=val_gen)
model.summary()
print("Evaluating validation set:")
model.evaluate(val_gen)
print("Evaluating test set : ")
model.evaluate(test_gen)