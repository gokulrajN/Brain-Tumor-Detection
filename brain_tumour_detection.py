from IPython.display import clear_output
!git clone https://github.com/atakanady/Mask_RCNN-_V1.git # Maske R-CNN kod uygulamasını yükle
!git clone https://github.com/atakanady/brain-tumor.git 
!pip install pycocotools #COCO, nesne algılama için tasarlanmış büyük bir görüntü veri kümesidir
#!rm -rf brain-tumor/.git/
#!rm -rf Mask_RCNN-_V1/.git/
clear_output()


import os 
import sys
from tqdm import tqdm
import cv2
import numpy as np
import json
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import random

# Projenin kök dizini;
ROOT_DIR = os.path.abspath('Mask_RCNN-_V1/')
sys.path.append(ROOT_DIR) 

from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))
import coco

plt.rcParams['figure.facecolor'] = 'white' #görsellerin arka planının renk seçimi.
clear_output()

def get_ax(rows=1, cols=1, size=7):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

MODEL_DIR = os.path.join(ROOT_DIR, 'logs') # Eğitimli modeli kaydetmek.
DATASET_DIR = 'brain-tumor/data_cleaned/' #Görüntü verilerini içeren dizin.
DEFAULT_LOGS_DIR = 'logs' 

# Eğitilmiş ağırlıklar dosya yolu.
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Eğitilmiş modelden bilgi alınamaz ise COCO eğitimli ağırlıkları sürümlerden indir.
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class TumorConfig(Config):
    
    # Yapılandırılmaya ad ekleniyor.
    NAME = 'tumor_detector'
    #Ne kadar GPU desteği sağlanacağını belirtiyoruz.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    #Sınıf sayısı (arka plan dahil)
    NUM_CLASSES = 1 + 1  # arkaplan + tümor
    
    # Daha hızlı eğitim için küçük resimler kullanın. Küçük tarafın sınırlarını belirler,
    # büyük taraf ve bu görüntü şeklini belirler.
    #IMAGE_MIN_DIM = 128
    #IMAGE_MAX_DIM = 128
    
    DETECTION_MIN_CONFIDENCE = 0.85    
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    
config = TumorConfig()
config.display()

class BrainScanDataset(utils.Dataset):
       # Şekiller sentetik veri kümesini oluşturur. Veri kümesi basitten oluşur
        #şekiller (üçgenler, kareler, daireler) boş bir yüzeye rastgele yerleştirilir.
        #Görüntüler anında oluşturulur. Dosya erişimi gerekmez.
    def load_brain_scan(self, dataset_dir, subset):
        
        #istenen sayıda sentetik görüntü üretin.
        #count: oluşturulacak görüntü sayısı.
        #yükseklik, genişlik: oluşturulan görüntülerin boyutu
        
        #class ekleniyor
        self.add_class("tumor", 1, "tumor")
        
        assert subset in ["train", "val", 'test']
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(DATASET_DIR, subset, 'annotations_'+subset+'.json')))
        annotations = list(annotations.values()) # dict anahtarlara gerek yok

        # VIA aracı, görüntüleri olmasa bile görüntüleri JSON'a kaydeder.
        annotations = [a for a in annotations if a['regions']]

        # Resim ekle
        for a in annotations:
            # X ,Y kordinatları alınır
            # x1 ve x2 versiyonlarını desteklemek için if fonk. gerekli.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask(), çokgenleri maskelere dönüştürmek için görüntü boyutuna ihtiyaç duyar.
            # Ne yazık ki, VIA onu JSON'a dahil etmiyor, bu yüzden okumalıyız.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "tumor",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, 
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):

        # farm_cow veri kümesi görüntüsü değilse, üst sınıfa yetki verin.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super(self.__class__, self).load_mask(image_id)
        
        # Çokgenleri bir bitmap şekil maskesine dönüştürün
        # [yükseklik, genişlik, örnek_sayısı]
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
        # Çokgenin içindeki piksel dizinlerini alın ve 1'e ayarlayın
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            
        # Döndürme maskesi ve her örneğin sınıf kimlikleri dizisi.Sahip olduğumuzdan beri yalnızca bir sınıf kimliği, 1'lik bir dizi döndürürüz.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        #Görüntü yolunu döndür
        
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


#Eğitim moduna model oluşturma.
model = modellib.MaskRCNN(
    mode='training', 
    config=config, 
    model_dir=DEFAULT_LOGS_DIR
)

model.load_weights(
    COCO_MODEL_PATH, 
    by_name=True, 
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)

#Eğitim veri Seti
dataset_train = BrainScanDataset()
dataset_train.load_brain_scan(DATASET_DIR, 'train')
dataset_train.prepare()

#Doğrulama Veri Seti
dataset_val = BrainScanDataset()
dataset_val.load_brain_scan(DATASET_DIR, 'val')
dataset_val.prepare()

#Test Veri Seti
dataset_test = BrainScanDataset()
dataset_test.load_brain_scan(DATASET_DIR, 'test')
dataset_test.prepare()

# Çok küçük bir veri seti kullandığımızdan ve COCO eğitimli ağırlıklar, çok uzun süre çalışmamıza gerek yok. Overfitting moduna girmemesi önemli.
print("Eğitim Ağı")
model.train(
    dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=15,
    layers='heads'
)
 
# Modeli çıkarım modunda yeniden oluşturun

model = modellib.MaskRCNN(
    mode="inference", 
    config=config,
    model_dir=DEFAULT_LOGS_DIR
)

# Kaydedilmiş ağırlıklara giden yolu alın
# Ya belirli bir yol belirleyin ya da son eğitilmiş ağırlıkları bulun
# model_path = os.path.join(ROOT_DIR, ".h5 dosya adı burada")
model_path = model.find_last()

# Load trained weights
print("Ağırlıklar yükleniyor", model_path)
model.load_weights(model_path, by_name=True)

def predict_and_plot_differences(dataset, img_id):
    original_image, image_meta, gt_class_id, gt_box, gt_mask =\
        modellib.load_image_gt(dataset, config, 
                               img_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]

    visualize.display_differences(
        original_image,
        gt_box, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        class_names = ['tumor'], title="", ax=get_ax(),
        show_mask=True, show_box=True)

def display_image(dataset, ind):
    plt.figure(figsize=(3,5))
    plt.imshow(dataset.load_image(ind))
    plt.xticks([])
    plt.yticks([])
    plt.title('Orjinal Resim')

ind = 0
display_image(dataset_val, ind)
predict_and_plot_differences(dataset_val, ind)

ind = 7
display_image(dataset_val, ind)
predict_and_plot_differences(dataset_val, ind)

ind = 4
display_image(dataset_val, ind)
predict_and_plot_differences(dataset_val, ind)

ind = 0
display_image(dataset_test, ind)
predict_and_plot_differences(dataset_test, ind)

ind = 3
display_image(dataset_test, ind)
predict_and_plot_differences(dataset_test, ind)

ind = 2
display_image(dataset_test, ind)
predict_and_plot_differences(dataset_test, ind)

ind = 0
display_image(dataset_test, ind)
predict_and_plot_differences(dataset_test, ind)
