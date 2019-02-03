import ssi_cnn as sc
import tensorflow as tf 


def get_model_autoencoder(lw1=1e-8):
    lips_inputs = tf.keras.Input(shape=(42, 50, 1), name='lips')
    #tongue_inputs = tf.keras.Input(shape=(42, 50, 1), name='tongue')
    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(lips_inputs)
    layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

    layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    encode = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(layer)
    
    layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(encode)
    layer = tf.keras.layers.UpSampling2D((2,2))(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
    
    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.UpSampling2D((2,2))(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.UpSampling2D((2,2))(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
    
    layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=tf.nn.relu)(layer)
    layer = tf.keras.layers.UpSampling2D((2,2))(layer)
    layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
    
    decode = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), kernel_regularizer=tf.keras.regularizers.l1(lw1), \
                                   padding='same', activation=None)(layer)

    
    optimizer=tf.keras.optimizers.Adam(lr = 1e-6, beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.0)
    
    model = tf.keras.Model(inputs = lips_inputs,outputs = decode)
    model.compile(optimizer=optimizer, loss = 'mse',metrics=["mse"])        
 
    return model


def keras_train(IS16=False):

    try:
        os.mkdir("../out/%s" % EXP_NAME)
    except:
        pass
    #load data
    train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf = sc.load_dataset(IS16)   
    
    #target preprocessing
    #Strain_ys, test_ys = sc.target_preprocessing(train_lsf,test_lsf,classification)
    
    #load model 
    model = get_model_autoencoder()
    print(model.summary())
    model.fit(train_tongue,train_tongue,verbose=2,batch_size=512,shuffle=True)
          
if __name__ == "__main__":
    keras_train(IS16=False)

    

