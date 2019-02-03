        
    
class datamanager():
    def __init__(self,lsf_path="../out/test_lsf/lsf.pkl",testrate=0.15):
        f = open(lsf_path,"rb")
        self.lsf = pickle.load(f)
        
        length = np.array([10670146,7050413,13645126,5311110,13410518])
        length = (length / 44100*60)
        test = length*testrate
        left = length.cumsum() - test
        right = length.cumsum()
        self.left = left.astype(int)
        self.right = right.astype(int)
        self.teststart = self.left[0]
        self.blockid = 0
        
            
    def reset_test(self):
        self.teststart = self.left[0]
        self.blockid = 0
    
            
        
    def get_batch(self,train,batchsize=512):

        lips  = []
        togues= []
        label = []

        for i in range(batchsize):
            
            if train:

                rand = np.random.randint(1,68147)

                while ((rand >= self.left[0]) & (rand <self.right[0]) | 
                       (rand >= self.left[1]) & (rand <self.right[1]) |
                       (rand >= self.left[2]) & (rand <self.right[2]) |
                       (rand >= self.left[3]) & (rand <self.right[3]) |
                       (rand >= self.left[4]) & (rand <self.right[4]) 
                       ):
                    rand = np.random.randint(1,68147)

            else:

                rand = self.teststart
                if ((rand >= self.left[0]) & (rand <self.right[0]) | 
                       (rand >= self.left[1]) & (rand <self.right[1]) |
                       (rand >= self.left[2]) & (rand <self.right[2]) |
                       (rand >= self.left[3]) & (rand <self.right[3]) |
                       (rand >= self.left[4]) & (rand <self.right[4]) 
                       ):
                    self.teststart += 1
                else:
                    if rand >= 68146:
                        break

                    self.blockid+=1
                    self.teststart =self.left[self.blockid]
                    rand = self.teststart
                    self.teststart += 1

            ran = str(rand)
            for i in range(6-len(ran)):
                ran = '0'+ran

            #fdir = "../data/Songs_Lips/%s.tif" % ran
            fdir = "../out/resize_lips/%s.tif" % ran

            lframe_0 = cv2.imread(fdir,0)
            lframe_0 = np.array(lframe_0)
        

            lips.append(lframe_0.reshape((lframe_0.shape[0],lframe_0.shape[1],1)))

            #fdir = "../data/Songs_Tongue/%s.tif" % ran
            fdir = "../out/resize_tongue/%s.tif" % ran
            tframe_0 = cv2.imread(fdir,0)
            tframe_0 = np.array(tframe_0)
            togues.append(tframe_0.reshape((tframe_0.shape[0],tframe_0.shape[1],1)))

            label.append(self.lsf[rand-1])
            
   
            
        return np.array(lips,dtype='float'),np.array(togues,dtype='float'), np.array(label,dtype='float')   


def simple_cnn_model(lip_l=42,lip_w=50,tog_l=42, tog_w=50,log = True):
    
  
    #input output format 
    lips = tf.placeholder(dtype='float32',shape=[None,lip_l,lip_w,1])
    togues = tf.placeholder(dtype='float32',shape=[None,tog_l,tog_w,1])
    y = tf.placeholder(dtype='float32',shape=[None,1])
    
    #conv network for lips and togues
    conv_lip = tf.layers.conv2d(inputs=lips,filters=16,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
    conv_lip = tf.layers.max_pooling2d(inputs = conv_lip, pool_size=(2,2),strides=(2,2),padding='same')
    conv_lip = tf.layers.batch_normalization(inputs=conv_lip,axis=1)
    #flatten of lips
    print(tf.shape(conv_lip))
    
    lip_flat = tf.layers.flatten(conv_lip)
    
    conv_tog = tf.layers.conv2d(inputs=togues,filters=16,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
    conv_tog = tf.layers.max_pooling2d(inputs = conv_tog, pool_size=(2,2),strides=(2,2),padding='same')
    conv_tog = tf.layers.batch_normalization(inputs=conv_tog,axis=1)
    #flatten of togues
    tog_flat = tf.layers.flatten(conv_lip)
    
    #concat of lips and togues
    concat_lip_tog = tf.concat([lip_flat,tog_flat],1)
    
    #predicted values as regression 
    pred = tf.layers.dense(concat_lip_tog,1,activation = None)
    
    #loss function 
    loss = tf.losses.mean_squared_error(labels=y,predictions = pred)
    if log:
        tf.summary.scalar('pred loss',loss)
    #optimizer 
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
    

    return lips,togues,y, optimizer, pred ,loss

def deep_cnn_model(lip_l=42,lip_w=50,tog_l=42, tog_w=50,log = True):
    
    #input output format 
    lips = tf.placeholder(dtype='float32',shape=[None,lip_l,lip_w,1])
    togues = tf.placeholder(dtype='float32',shape=[None,tog_l,tog_w,1])
    y = tf.placeholder(dtype='float32',shape=[None,1])
    
    #conv network for lips and togues
    conv_lip = tf.layers.conv2d(inputs=lips,filters=16,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
    conv_lip = tf.layers.max_pooling2d(inputs = conv_lip, pool_size=(2,2),strides=(2,2),padding='same')
    conv_lip = tf.layers.batch_normalization(inputs=conv_lip,axis=1)
    
#     conv_lip = tf.layers.conv2d(inputs=lips,filters=32,kernel_size=(3,3),padding="same",activation=tf.nn.relu)
#     conv_lip = tf.layers.max_pooling2d(inputs = conv_lip, pool_size=(2,2),strides=(2,2),padding='same')
#     conv_lip = tf.layers.batch_normalization(inputs=conv_lip,axis=1)
    
#     conv_lip = tf.layers.conv2d(inputs=lips,filters=32,kernel_size=(3,3),padding="same",activation=tf.nn.relu)
#     conv_lip = tf.layers.max_pooling2d(inputs = conv_lip, pool_size=(2,2),strides=(2,2),padding='same')
#     conv_lip = tf.layers.batch_normalization(inputs=conv_lip,axis=1)
    
    #flatten of lips
    lip_flat = tf.layers.flatten(conv_lip)
    
    conv_tog = tf.layers.conv2d(inputs=togues,filters=16,kernel_size=(5,5),padding="same",activation=tf.nn.relu)
    conv_tog = tf.layers.max_pooling2d(inputs = conv_tog, pool_size=(2,2),strides=(2,2),padding='same')
    conv_tog = tf.layers.batch_normalization(inputs=conv_tog,axis=1)
    
#     conv_tog = tf.layers.conv2d(inputs=togues,filters=32,kernel_size=(3,3),padding="same",activation=tf.nn.relu)
#     conv_tog = tf.layers.max_pooling2d(inputs = conv_tog, pool_size=(2,2),strides=(2,2),padding='same')
#     conv_tog = tf.layers.batch_normalization(inputs=conv_tog,axis=1)
    
#     conv_tog = tf.layers.conv2d(inputs=togues,filters=32,kernel_size=(3,3),padding="same",activation=tf.nn.relu)
#     conv_tog = tf.layers.max_pooling2d(inputs = conv_tog, pool_size=(2,2),strides=(2,2),padding='same')
#     conv_tog = tf.layers.batch_normalization(inputs=conv_tog,axis=1)
    
    #flatten of togues
    tog_flat = tf.layers.flatten(conv_lip)
    
    #concat of lips and togues
    concat_lip_tog = tf.concat([lip_flat,tog_flat],1)
    concat_length = concat_lip_tog.shape[1].value
    print(concat_length)
    #predicted values as regression 
    concat_lip_tog = tf.layers.dense(concat_lip_tog,int(concat_length/2),activation = tf.nn.relu)
#     concat_lip_tog = tf.layers.dense(concat_lip_tog,int(concat_length/2/2),activation = tf.nn.relu)
    pred = tf.layers.dense(concat_lip_tog,1,activation = None)
    
    #loss function 
    loss = tf.losses.mean_squared_error(labels=y,predictions = pred)
    if log:
        tf.summary.scalar('pred loss',loss)
    #optimizer 
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
    

    return lips,togues,y, optimizer, pred ,loss


def train():
                
    dm = datamanager(lsf_path='../out/test_lsf/lsf.pkl')
 
    try:
        os.mkdir("../out/190125_deep_cnn/")
    except:
        pass
    with tf.Session() as sess:
        lips,togues,y, optimizer, pred ,loss = deep_cnn_model()
        init = tf.global_variables_initializer()
        sess.run(init)
        merged = tf.summary.merge_all()
        trainwriter = tf.summary.FileWriter("logs/train"  ,sess.graph)
        testwriter  = tf.summary.FileWriter("logs/test"  ,sess.graph)

        for i in tqdm(range(10000)):
            lips_batch, togues_batch, label_batch = dm.get_batch(train=True,batchsize=32)
            print("batch")
            
            rs = sess.run(pred,feed_dict=({lips:lips_batch,togues:togues_batch,y : label_batch[:,0][:,np.newaxis]}))
            print(rs)
            rs = sess.run(optimizer,feed_dict=({lips:lips_batch,togues:togues_batch,y : label_batch[:,0][:,np.newaxis]}))
            
            if i % 100 == 0:
                rs = sess.run(loss,feed_dict=({lips:lips_batch,togues:togues_batch,y : label_batch[:,0][:,np.newaxis]}))
                print("epochs %d train loss %.6f"% (i, rs))
                dm.reset_test()
                results = []
                real = []
                for _ in range(10*4):
                    lips_test_batch, togues_test_batch, label_test_batch = dm.get_batch(train=False,batchsize=64)
                    #rs = sess.run(loss,feed_dict=({lips:lips_test_batch,togues:togues_test_batch,\
                    #                               y : label_test_batch[:,0][:,np.newaxis]}))
                    rs = sess.run(pred,feed_dict=({lips:lips_test_batch,togues:togues_test_batch,\
                                                   y : label_test_batch[:,0][:,np.newaxis]}))
                    
                    real.extend(list(label_test_batch[:,0]))
                    results.extend(list(rs.reshape(-1)))
                    
                
                plt.title("LSF1")
                plt.plot(range(len(real)),real,label='real LSF',linewidth=1)
                plt.plot(range(len(results)),results,label='predicted LSF',linewidth=1)
                plt.legend()
                plt.savefig("../out/190125_deep_cnn/simpletest_epoch%d.png" % i)
                plt.close()
                trainrs = sess.run(merged,feed_dict={lips:lips_batch,togues:togues_batch,y:label_batch[:,0][:,np.newaxis]})
                trainwriter.add_summary(trainrs,i)