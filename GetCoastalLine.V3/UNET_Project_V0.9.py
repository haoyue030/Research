#!/usr/bin/env python
# coding: utf-8

#!pip install tensorflow
#!pip install openpyxl
#!pip install opencv-python

import time
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime 
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Rescaling
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import models, layers, regularizers # type: ignore
from tensorflow.keras import backend as K # type: ignore
# from keras.utils import normalize # type: ignore
from tensorflow.keras.models import Model # type: ignore
from keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
from keras import layers, Model, initializers
from keras.layers import Input, Rescaling, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Activation
from keras.layers import Concatenate  # 用類別而非函式別名，避免沒 import 造成 NameError
from keras.saving import register_keras_serializable
import tensorflow as tf


# ---- Attention Gate（穩定版）----
import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, Lambda
)
@register_keras_serializable()
class PsiShaper(layers.Layer):
    """等同於：psi = floor + (1-floor) * sigmoid(preact / tau)。"""
    def __init__(self, tau=1.0, floor=0.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.floor = float(floor)

    def call(self, preact):
        x = preact / self.tau if self.tau and self.tau != 1.0 else preact
        psi = tf.nn.sigmoid(x)
        if self.floor and self.floor > 0:
            psi = self.floor + (1.0 - self.floor) * psi
        return psi

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"tau": self.tau, "floor": self.floor})
        return cfg

@register_keras_serializable()
class DebugTap(layers.Layer):
    """可選的 debug 列印（序列化安全）。"""
    def __init__(self, gate_id="", enable=False, **kwargs):
        super().__init__(**kwargs)
        self.gate_id = gate_id
        self.enable = bool(enable)

    def call(self, t):
        if self.enable:
            tf.print(f"[AG-{self.gate_id}] psi min/max/mean:",
                     tf.reduce_min(t), tf.reduce_max(t), tf.reduce_mean(t))
        return t

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gate_id": self.gate_id, "enable": self.enable})
        return cfg
    
def attention_gate(x, g, inter_channels, gate_id="",
                   tau=1.05,        # 你原本的建議值
                   bias_init=-0.2,
                   floor=0.10,
                   debug=False):
    he = initializers.HeNormal()
    bias = initializers.Constant(bias_init)

    theta_x = Conv2D(inter_channels, 1, padding='same', use_bias=True,
                     kernel_initializer=he, name=f"ag{gate_id}_theta_x")(x)
    phi_g   = Conv2D(inter_channels, 1, padding='same', use_bias=True,
                     kernel_initializer=he, name=f"ag{gate_id}_phi_g")(g)
    act     = Activation('relu', name=f"ag{gate_id}_relu")(layers.Add(name=f"ag{gate_id}_add")([theta_x, phi_g]))

    # 這裡不再用 Lambda：先做 1x1 conv，再用可序列化的 PsiShaper 完成 tau 與 floor
    preact  = Conv2D(1, 1, padding='same', use_bias=True,
                     kernel_initializer=he, bias_initializer=bias,
                     name=f"ag{gate_id}_psi_conv")(act)
    psi     = PsiShaper(tau=tau, floor=floor, name=f"ag{gate_id}_psi")(preact)
    psi     = DebugTap(gate_id=str(gate_id), enable=bool(debug), name=f"ag{gate_id}_dbg")(psi)

    out = layers.Multiply(name=f'gate_mul_{gate_id}')([x, psi])
    return out

def attention_unet(n_classes=5, IMG_HEIGHT=896, IMG_WIDTH=512, IMG_CHANNELS=3, Activation='relu', use_attention=True):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    dropoutRate = 0.3
    s = Rescaling(1./255, dtype='float32')(inputs)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation=Activation, kernel_initializer='RandomNormal', padding='same')(s)
    c1 = Dropout(dropoutRate)(c1)
    c1 = Conv2D(16, (3, 3), activation=Activation, kernel_initializer='RandomNormal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropoutRate)(c2)
    c2 = Conv2D(32, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropoutRate)(c3)
    c3 = Conv2D(64, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropoutRate)(c4)
    c4 = Conv2D(128, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c5)

    # inter_channels = skip 的一半（至少 1）
    def _ic(tensor):
        ch = int(tensor.shape[-1])
        return max(ch // 2, 1)

    # Decoder + Attention
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    if use_attention:
        a6 = attention_gate(c4, u6, _ic(c4), gate_id="6", debug=False, floor=0.10, bias_init=-0.30, tau=0.95)
        u6 = Concatenate(axis=3)([u6, a6])
    c6 = Conv2D(128, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(dropoutRate)(c6)
    c6 = Conv2D(128, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    if use_attention:
        a7 = attention_gate(c3, u7, _ic(c3), gate_id="7", debug=False, bias_init=-0.25, tau=0.95)
        u7 = Concatenate(axis=3)([u7, a7])
    c7 = Conv2D(64, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(dropoutRate)(c7)
    c7 = Conv2D(64, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    if use_attention:
        a8 = attention_gate(c2, u8, _ic(c2), gate_id="8", debug=False, floor=0.15, bias_init=-0.10, tau=1.05)
        u8 = Concatenate(axis=3)([u8, a8])
    c8 = Conv2D(32, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(dropoutRate)(c8)
    c8 = Conv2D(32, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    if use_attention:
        a9 = attention_gate(c1, u9, _ic(c1), gate_id="9", debug=False, floor=0.10, bias_init=-0.25, tau=1.00)
        u9 = Concatenate(axis=3)([u9, a9])
    c9 = Conv2D(16, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(dropoutRate)(c9)
    c9 = Conv2D(16, (3, 3), activation=Activation, kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# In[26]:
'''
# 1. 資料準備
'''
SIZE_X = 896 
SIZE_Y = 512
n_classes=6 #Number of classes for segmentation

#Capture training image info as a list
train_images = []
imfilenames = []
for directory_path in glob.glob("image/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        imfilenames.append(os.path.basename(img_path))
        img = cv2.imread(img_path) 
        # drop the last column and first row to make the num of column and row to 600 and 960
        # cropped_img = img[1:, :-1]
        cropped_img = img[30:926, :512]
        train_images.append(cropped_img)
#print(cropped_img.shape)       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)
nI = train_images.shape[0]

print(train_images.shape)

# In[27]:


#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("label/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        #cropped_mask = mask[1:, :-1]
        cropped_mask = mask[30:926, :512] # make the image to the size of 896 * 512
        train_masks.append(cropped_mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)


# In[8]:


print(train_masks.shape)


# # we randomly adjust brightness of the training images
# 
# brightened_images = []
# 
# for i in range(nI ):
#     a = (np.random.rand(1)-0.5)/2
#     image = train_images[i]
#     image = tf.image.adjust_brightness(image, delta=a)
#     brightened_images.append(image)
#     cv2.imwrite(os.path.join('adjust_Brightness_images/',imfilenames[i]),image.numpy())
# brightened_images = np.array(brightened_images)
# print(brightened_images.shape)

# In[28]:


#Capture training image info as a list
brightened_images = []
for directory_path in glob.glob("adjust_Brightness_images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path) 
        
        brightened_images.append(img)
#print(cropped_img.shape)       
#Convert list to array for machine learning processing        
brightened_images = np.array(brightened_images)


# # we randomly adjust saturation of the training images
# 
# adjustSaturation_images = []
# 
# for i in range(nI ):
#     a = 1+(np.random.rand(1)-0.5)/2
#     
#     image = train_images[i]
#     image = tf.image.adjust_saturation(image, a[0])
#     adjustSaturation_images.append(image)
#     cv2.imwrite(os.path.join('adjust_Saturation_images/',imfilenames[i]),image.numpy())
#     
# adjustSaturation_images = np.array(adjustSaturation_images)
# print(adjustSaturation_images.shape)
# #print(adjustSaturation_images[0])

# In[29]:


#Capture training image info as a list
adjustSaturation_images = []
for directory_path in glob.glob("adjust_Saturation_images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path) 
        
        adjustSaturation_images.append(img)
#print(cropped_img.shape)       
#Convert list to array for machine learning processing        
adjustSaturation_images = np.array(adjustSaturation_images)


# In[78]:


# we randomly adjust saturation and also brightness of the training images

adjustSaturationNBrightness_images = []

for i in range(nI ):
    a = 1+(np.random.rand(1)-0.5)/2
    b = (np.random.rand(1)-0.5)/2
    image = train_images[i]
    image = tf.image.adjust_saturation(image, a[0])
    image = tf.image.adjust_brightness(image, delta=b)
    adjustSaturationNBrightness_images.append(image)

adjustSaturationNBrightness_images = np.array(adjustSaturationNBrightness_images)
print(adjustSaturationNBrightness_images.shape)


# In[12]:


def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)


# In[104]:


#noi = 15
#visualize(train_images[noi], adjustSaturationNBrightness_images[noi])


# In[105]:


#noi = 67
#visualize(train_images[noi], brightened_images[noi])


# In[30]:


#argumented_images = np.concatenate([train_images,brightened_images,adjustSaturation_images,adjustSaturationNBrightness_images], axis=0)
argumented_images = np.concatenate([train_images,brightened_images,adjustSaturation_images], axis=0)
print(argumented_images.shape)


# In[31]:


argumented_masks = np.concatenate([train_masks, train_masks,train_masks], axis=0)
print(argumented_masks.shape)


# In[32]:


train_masks = argumented_masks
train_images = argumented_images


# In[16]:


###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
print(n,h,w)
train_masks_reshaped = train_masks.reshape(-1,1)
print(train_masks_reshaped.shape)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)

train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
np.unique(train_masks_encoded_original_shape)
#print(train_masks_encoded_original_shape.shape)


# In[17]:


print(train_images.shape)


# In[18]:


print(train_masks_encoded_original_shape.shape)


# In[19]:


#################################################
train_images_expanded = np.expand_dims(train_images, axis=3)
print(train_images_expanded.shape)

#train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
print(train_masks_input.shape)


# In[20]:


#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 
print(y_train.shape)
print(X_train.shape)
print(y_test.shape)
print(X_test.shape)
    

# In[21]:


#print(n_classes)
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))


test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

print(X_train.shape)


# In[22]:
'''
# 2. 模式選擇
'''
def get_model():
    return attention_unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() 


# In[23]:
'''
# 3. 模式訓練
'''
start_time = time.time()
history = model.fit(X_train, y_train_cat, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=25, 
                    validation_data=(X_test, y_test_cat), 
                    #class_weight=class_weights,
                    shuffle=True)
                    
end_time = time.time()

 # 計算並顯示執行時間
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time} 秒")



# In[108]:


current_date = datetime.today()
formatted_date = current_date.strftime("%Y%m%d")
print(formatted_date)


# In[122]:


# 檔名設定
outexcelfile = 'UNET_model_history_'+formatted_date+'_'+'V0.xlsx'
outkerasfile = 'UNET_'+formatted_date+'_'+'_'+str(253)+'.weights.h5'   # 👉 純權重 .h5
outkeras_full = 'UNET_'+formatted_date+'_'+'_'+str(253)+'.keras'       # 👉 整模 .keras（保險）
print(outexcelfile)
v = int(outexcelfile[-6:-5])

# 版本遞增
if os.path.exists(outexcelfile):
    v = int(outexcelfile[-6:-5]) + 1
    outexcelfile = 'UNET_model_history_'+formatted_date+'_'+str(v)+'.xlsx'
    outkerasfile = 'UNET_'+formatted_date+'_'+str(v)+'-'+str(253)+'.weights.h5'  # 👉 改成 .weights.h5
    outkeras_full = 'UNET_'+formatted_date+'_'+str(v)+'-'+str(253)+'.keras'      # 👉 同步給 .keras
    print(outexcelfile+' will be saved!')
    print(outkerasfile+' will be saved!')
    print(outkeras_full+' will be saved!')
else:
    print(outexcelfile+' will be saved!')
    print(outkerasfile+' will be saved!')
    print(outkeras_full+' will be saved!')

# ======= 訓練完後的儲存動作 =======
# 純權重（.h5）— 之後推論用 build_unet(...) + load_weights(...)
model.save_weights(outkerasfile)

# 整模（.keras）— 保險用，跨環境最穩（可選）
model.save(outkeras_full)

    


# In[124]:


############################################################
'''
# 4. 模式表現檢查
'''
#Evaluate the model
	# evaluate model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


df_history = pd.DataFrame({'loss': loss, 'val_loss': val_loss,'acc': acc, 'val_acc': val_acc})

df_history.to_excel(outexcelfile)
model.save(outkerasfile)
print('model saved!')

###
#plot the training and validation accuracy and loss at each epoch

epochs = range(1, len(loss) + 1)
plt.rcParams.update({'font.family': 'Microsoft JhengHei', 'font.size': 16})
plt.plot(epochs, loss, 'yo-', label='Training', markersize=5,color='#4f6b8d')  # 'o-' adds a circle marker
plt.plot(epochs, val_loss, 'ro-', label='Validation', markersize=5,color='#cf3832')  # 'o-' adds a circle marker
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize='large')  # Set the legend font size to large
plt.grid(linestyle='--')
plt.tight_layout(pad=1.0)
plt.savefig('UNET_loss_20240327.tif', dpi=300, format='tiff')
# plt.show()
plt.close()

plt.rcParams.update({'font.family': 'Microsoft JhengHei', 'font.size': 16})
plt.plot(epochs, acc, 'yo-', label='Training', markersize=5,color='#4f6b8d')  # 'o-' adds a circle marker
plt.plot(epochs, val_acc, 'ro-', label='Validation', markersize=5,color='#cf3832')  # 'o-' adds a circle marker
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(fontsize='large')  # Set the legend font size to large
plt.grid(linestyle='--')
plt.tight_layout(pad=1.0)
plt.savefig('UNET_Accuracy_20240327.tif', dpi=300, format='tiff')
# plt.show()
plt.close()



# In[99]:
'''
# load saved model
from keras.models import load_model

model = load_model('UNET_NEW_Image20240331_1.keras',safe_mode=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_, acc = model.evaluate(X_test,y_test_cat)
print('Accuracy = ',(acc*100.0), '%')
'''

# In[125]:
'''
# 5. 計算分類準確度
'''

y_pred=model.predict(X_test)


# In[126]:


print(y_pred.shape)
print(y_test.shape)
print(X_test.shape)


# for test

testImgsize = X_test.shape[0]

# calculate confusion matrix
prediction_test = np.argmax(y_pred, axis=3)
labelled = y_test[:,:,:,0]
cmM = []

for i in range(testImgsize):
    labelImg = labelled[i,:,:]
    predictImg = prediction_test[i,:,:] 
    #print(labelImg.shape)
    y_true_flat = labelImg.reshape(-1)
    y_pred_flat = predictImg.reshape(-1)
    #print(np.max(y_pred_flat))
    cm = tf.math.confusion_matrix(labels=y_true_flat, predictions=y_pred_flat, num_classes=6)
    #print(cm)
    cmM.append(cm)
    #


labels = ['沙灘', '海水', '礁岩', '植生', '建築', '無']

cm_Array = np.array(cmM)
#print(cm_Array.shape)
avg_cm = np.mean(cm_Array,axis=0)
np.set_printoptions(formatter={'all': lambda x: f'{x:0.2f}'})
#print(avg_cm)

sum_totalsamples = np.sum(avg_cm,axis=0)
avg_cmRatio=[]
for i,total in enumerate(sum_totalsamples):
    #print(i,total)
    #print(avg_cm[:,i])
    ratio = avg_cm[:,i]/total;
    
    avg_cmRatio.append(ratio)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
avg_cmRatio = np.array(avg_cmRatio)
plt.imshow(avg_cmRatio.T, cmap='Blues', interpolation='nearest')
for i in range(avg_cmRatio.shape[0]):
    for j in range(avg_cmRatio.shape[1]):
        plt.text(i,j, f'{avg_cmRatio[i, j]:.3f}', ha='center', va='center', color='#da4f3e',fontsize=14, weight='bold')

plt.colorbar()
#plt.title('Average Confusion Matrix')
plt.tick_params(axis='x', which='both', bottom=False, top=True)
plt.xlabel('實際類別')
plt.ylabel('分類結果')
plt.xticks(range(avg_cm.shape[1]), labels, rotation=0, ha='center', position=(0,1.1))
plt.yticks(range(avg_cm.shape[0]), labels)
plt.savefig('AverageConfusionMatrix.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


num_classes = cm.shape[0]
overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

producer_accuracy = np.zeros(num_classes)
user_accuracy = np.zeros(num_classes)

for i in range(num_classes):
    producer_accuracy[i] = cm[i, i] / np.sum(cm[i, :])
    user_accuracy[i] = cm[i, i] / np.sum(cm[:, i])
print("for Testing:")
print(f"Overall Accuracy: {overall_accuracy:.3f}")
print("Producer's Accuracy:", ", ".join([f"{acc:.3f}" for acc in producer_accuracy]))
print("User's Accuracy:", ", ".join([f"{acc:.3f}" for acc in user_accuracy]))




# ======================================================
# for training
y_pred_train=model.predict(X_train)
trainImgsize = X_train.shape[0]

# calculate confusion matrix
prediction = np.argmax(y_pred_train, axis=3)
labelled = y_train[:,:,:,0]
cmM = []

for i in range(trainImgsize):
    labelImg = labelled[i,:,:]
    predictImg = prediction[i,:,:] 
    #print(labelImg.shape)
    y_true_flat = labelImg.reshape(-1)
    y_pred_flat = predictImg.reshape(-1)
    #print(np.max(y_pred_flat))
    cm = tf.math.confusion_matrix(labels=y_true_flat, predictions=y_pred_flat, num_classes=6)
    #print(cm)
    cmM.append(cm)
    #


cm_Array = np.array(cmM)
#print(cm_Array.shape)
avg_cm = np.mean(cm_Array,axis=0)
np.set_printoptions(formatter={'all': lambda x: f'{x:0.2f}'})
#print(avg_cm)

sum_totalsamples = np.sum(avg_cm,axis=0)
avg_cmRatio=[]
for i,total in enumerate(sum_totalsamples):
    #print(i,total)
    #print(avg_cm[:,i])
    ratio = avg_cm[:,i]/total;
    
    avg_cmRatio.append(ratio)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
avg_cmRatio = np.array(avg_cmRatio)
plt.imshow(avg_cmRatio.T, cmap='Blues', interpolation='nearest')
for i in range(avg_cmRatio.shape[0]):
    for j in range(avg_cmRatio.shape[1]):
        plt.text(i,j, f'{avg_cmRatio[i, j]:.3f}', ha='center', va='center', color='#da4f3e',fontsize=14, weight='bold')

plt.colorbar()
#plt.title('Average Confusion Matrix')
plt.tick_params(axis='x', which='both', bottom=False, top=True)
plt.xlabel('實際類別')
plt.ylabel('分類結果')
plt.xticks(range(avg_cm.shape[1]), labels, rotation=0, ha='center', position=(0,1.1))
plt.yticks(range(avg_cm.shape[0]), labels)
plt.savefig('AverageConfusionMatrix_train.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


num_classes = cm.shape[0]
overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

producer_accuracy = np.zeros(num_classes)
user_accuracy = np.zeros(num_classes)

for i in range(num_classes):
    producer_accuracy[i] = cm[i, i] / np.sum(cm[i, :])
    user_accuracy[i] = cm[i, i] / np.sum(cm[:, i])
print("for Training:")
print(f"Overall Accuracy: {overall_accuracy:.3f}")
print("Producer's Accuracy:", ", ".join([f"{acc:.3f}" for acc in producer_accuracy]))
print("User's Accuracy:", ", ".join([f"{acc:.3f}" for acc in user_accuracy]))






# 先建立輸出資料夾（避免 OSError）
for _d in ['UNET_CoastLine_test', 'UNET_CoastLine_train', 'UNET_Classified_train', 'UNET Classified_text']:
    os.makedirs(_d, exist_ok=True)

imfilenames = imfilenames * 3

# ====== 測試集 ======
noi = y_test.shape[0]
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

for i in range(noi):
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_test[i, :, :])
    plt.axis('off')
    plt.title('UNET分類結果', fontsize=16)

    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i, :, :, 0])  # 改成 2D
    plt.axis('off')
    plt.title('人工標註類別', fontsize=16)

    plt.subplot(1, 3, 1)
    rgbImg = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2RGB)
    plt.imshow(rgbImg)
    plt.axis('off')
    plt.title('原始影像', fontsize=16)

    # 邊界（確保 2D）
    edges = cv2.Canny((y_test[i, :, :, 0] == 1).astype(np.uint8), 0, 1)
    plt.contour(edges, colors='red', linewidths=0.5, linestyles='dotted')

    edges_pred = cv2.Canny((prediction_test[i, :, :] == 1).astype(np.uint8), 0, 1)
    plt.contour(edges_pred, colors='blue', linewidths=0.5, linestyles='dotted')

    # 座標與輸出
    xul67, yul67 = 342800, 2771160
    indices = np.argwhere(edges == 255)
    Bd67X = xul67 + indices[:, 1]
    Bd67Y = yul67 - indices[:, 0]
    Bd = pd.DataFrame(np.vstack((Bd67X, Bd67Y)).T)

    indices = np.argwhere(edges_pred == 255)
    Bd67X = xul67 + indices[:, 1]
    Bd67Y = yul67 - indices[:, 0]
    Bd_UNET = pd.DataFrame(np.vstack((Bd67X, Bd67Y)).T)

    base = os.path.splitext(imfilenames[i])[0]  # 正確去副檔名
    outBd_path = os.path.join('UNET_CoastLine_test', f'Labeled_{base}.xlsx')
    Bd.to_excel(outBd_path)
    outBd_UNET_path = os.path.join('UNET_CoastLine_test', f'UNET_{base}.xlsx')
    Bd_UNET.to_excel(outBd_UNET_path)

    # 圖例
    x, y, l, h = 250, 50, 250, 100
    legend_rect = FancyBboxPatch((x, y), l, h, facecolor='white', boxstyle="round")
    plt.gca().add_patch(legend_rect)
    plt.gca().add_line(Line2D([x+10, x+50], [y+30, y+30], color='red', linestyle='dotted', linewidth=2))
    plt.text(x+65, y+30, '人工標註', verticalalignment='center', fontsize=10)
    plt.gca().add_line(Line2D([x+10, x+50], [y+70, y+70], color='blue', linestyle='dotted', linewidth=2))
    plt.text(x+65, y+70, 'UNET分類結果', verticalalignment='center', fontsize=10)

    outfigpath = os.path.join('UNET Classified_text', imfilenames[i])  # 保留你的原資料夾名稱
    plt.savefig(outfigpath, dpi=300, format='tif')
    plt.close()

# ====== 訓練集 ======
noi = y_train.shape[0]

for i in range(noi):
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 3)
    plt.imshow(prediction[i, :, :])
    plt.axis('off')
    plt.title('UNET分類結果', fontsize=16)

    plt.subplot(1, 3, 2)
    plt.imshow(y_train[i, :, :, 0])  # 改成 2D
    plt.axis('off')
    plt.title('人工標註類別', fontsize=16)

    plt.subplot(1, 3, 1)
    rgbImg = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
    plt.imshow(rgbImg)
    plt.axis('off')
    plt.title('原始影像', fontsize=16)

    edges = cv2.Canny((y_train[i, :, :, 0] == 1).astype(np.uint8), 0, 1)
    plt.contour(edges, colors='red', linewidths=0.5, linestyles='dotted')

    edges_pred = cv2.Canny((prediction[i, :, :] == 1).astype(np.uint8), 0, 1)
    plt.contour(edges_pred, colors='blue', linewidths=0.5, linestyles='dotted')

    xul67, yul67 = 342800, 2771160
    indices = np.argwhere(edges == 255)
    Bd67X = xul67 + indices[:, 1]
    Bd67Y = yul67 - indices[:, 0]
    Bd = pd.DataFrame(np.vstack((Bd67X, Bd67Y)).T)

    indices = np.argwhere(edges_pred == 255)
    Bd67X = xul67 + indices[:, 1]
    Bd67Y = yul67 - indices[:, 0]
    Bd_UNET = pd.DataFrame(np.vstack((Bd67X, Bd67Y)).T)

    base = os.path.splitext(imfilenames[i])[0]  # 正確去副檔名
    outBd_path = os.path.join('UNET_CoastLine_train', f'Labeled_{base}.xlsx')
    Bd.to_excel(outBd_path)

    outBd_UNET_path = os.path.join('UNET_CoastLine_train', f'UNET_{base}.xlsx')
    Bd_UNET.to_excel(outBd_UNET_path)

    x, y, l, h = 250, 50, 250, 100
    legend_rect = FancyBboxPatch((x, y), l, h, facecolor='white', boxstyle="round")
    plt.gca().add_patch(legend_rect)
    plt.gca().add_line(Line2D([x+10, x+50], [y+30, y+30], color='red', linestyle='dotted', linewidth=2))
    plt.text(x+65, y+30, '人工標註', verticalalignment='center', fontsize=10)
    plt.gca().add_line(Line2D([x+10, x+50], [y+70, y+70], color='blue', linestyle='dotted', linewidth=2))
    plt.text(x+65, y+70, 'UNET分類結果', verticalalignment='center', fontsize=10)

    outfigpath = os.path.join('UNET_Classified_train', imfilenames[i])
    plt.savefig(outfigpath, dpi=300, format='tif')
    plt.close()





























# # In[127]:


# # calculate confusion matrix
# prediction = y_pred_argmax=np.argmax(y_pred, axis=3)
# labelled = y_test[:,:,:,0]
# cmM = []

# for i in range(30):
#     labelImg = labelled[i,:,:]
#     predictImg = prediction[i,:,:] 
#     print(labelImg.shape)
#     y_true_flat = labelImg.reshape(-1)
#     y_pred_flat = predictImg.reshape(-1)
#     print(np.max(y_pred_flat))
#     cm = tf.math.confusion_matrix(labels=y_true_flat, predictions=y_pred_flat, num_classes=6)
#     print(cm)
    
#     cmM.append(cm.numpy())
#     #
# print(cmM)

# df = pd.DataFrame(cmM)

# # Write the DataFrame to an Excel file
# df.to_excel('confusionMatrix.xlsx', index=False)


# # In[134]:


# num_classes = cm.shape[0]
# overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# producer_accuracy = np.zeros(num_classes)
# user_accuracy = np.zeros(num_classes)

# for i in range(num_classes):
#     producer_accuracy[i] = cm[i, i] / np.sum(cm[i, :])
#     user_accuracy[i] = cm[i, i] / np.sum(cm[:, i])

# print(f"Overall Accuracy: {overall_accuracy:.2f}")
# print("Producer's Accuracy:", ", ".join([f"{acc:.2f}" for acc in producer_accuracy]))
# print("User's Accuracy:", ", ".join([f"{acc:.2f}" for acc in user_accuracy]))


# # In[130]:


# noi = y_test.shape[0]
# print(noi)


# # In[132]:


# for i in range(noi):
#     plt.figure()
#     plt.subplot(1, 3, 3)
#     plt.imshow(y_pred_argmax[i,:,:])
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(y_test[i])
#     plt.axis('off')
#     plt.subplot(1, 3, 1)
#     rgbImg = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2RGB)
#     plt.imshow(rgbImg)
#     plt.axis('off')
#     outfigpath = os.path.join('classificationResults/',imfilenames[i])
#     plt.savefig(outfigpath, dpi=300, format='tiff')
#     plt.show()

# # In[41]:


# #IOU

# #print(y_pred.shape)
# #y_pred_argmax=np.argmax(y_pred, axis=3)
# plt.figure()
# plt.subplot(1, 3, 3)
# plt.imshow(y_pred_argmax[1,:,:])
# plt.subplot(1, 3, 2)
# plt.imshow(train_masks[1])
# plt.subplot(1, 3, 1)
# plt.imshow(X_test[1])
# ##################################################

# #Using built in keras function
# from keras.metrics import MeanIoU
# n_classes = 6
# IOU_keras = MeanIoU(num_classes=n_classes)  

# IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())


# #To calculate I0U for each class...
# values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
# class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

# print("IoU for class1 is: ", class1_IoU)
# print("IoU for class2 is: ", class2_IoU)
# print("IoU for class3 is: ", class3_IoU)
# print("IoU for class4 is: ", class4_IoU)

# #plt.imshow(train_images[0, :,:,0], cmap='gray')
# #plt.imshow(train_masks[0], cmap='gray')


# # In[47]:


# print(IOU_keras.result())


# # In[84]:


# label = train_masks[1,:,:]

# print(label.shape)
# plt.imshow(label)


# # In[85]:


# cropped_label = label[30:926, :512] 
# print(cropped_label.shape)
# plt.imshow(cropped_label)


# # In[83]:


# img = train_images[1,:,:,:]
# #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img.squeeze()  # Remove the extra dimension
# print(img.shape)
# plt.imshow(img)


# # In[ ]:


# cropped_img = img[1:, :-1]


# In[ ]:




