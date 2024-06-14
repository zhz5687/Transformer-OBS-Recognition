import sys

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
import sys
import numpy as np
from torch import true_divide

from src.ui.obs_gui import *
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from skimage import transform, io, img_as_float
from skimage.util import pad

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json

class Residual(nn.Module):
    def __init__(self, input_channels, min_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, min_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(min_channels, min_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.Conv2d(min_channels, num_channels,
                               kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(min_channels)
        self.bn2 = nn.BatchNorm2d(min_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)



def resnet_block(input_channels, min_channels, num_channels, num_residuals, stride,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, min_channels, num_channels,
                                use_1x1conv=True, strides=stride))
        elif first_block and i == 0:
            blk.append(Residual(input_channels, min_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, min_channels, num_channels))
    return blk

class TestData(Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        with open('Validation_test.json', 'r', encoding='utf8') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'].replace('\\','/'))
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        width, height = image.size
        if width>height:
            dy = width - height

            yl = round(dy / 2)
            yr = dy - yl
            train_transform = transforms.Compose([
                transforms.Pad([0, yl, 0, yr], fill=(255, 255, 255), padding_mode='constant'),
                ])
        else:
            dx = height - width
            xl = round(dx / 2)
            xr = dx - xl
            train_transform = transforms.Compose([
                transforms.Pad([xl, 0, xr, 0], fill=(255, 255, 255), padding_mode='constant'),
                ])

        image = train_transform(image)
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.85233593, 0.85246795, 0.8517555], [0.31232414, 0.3122127, 0.31273854])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label,self.images[item]['path'].replace('\\','/')

    def __len__(self):
        return len(self.images)

class ScribbleArea(QFrame):  #

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        #resize设置宽高，move设置位置
        self.resize(250, 257)
        # self.move(100, 100)
        # self.setWindowTitle("简单的画板4.0")

        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)
        '''
            要想将按住鼠标后移动的轨迹保留在窗体上
            需要一个列表来保存所有移动过的点
        '''

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 6
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()

    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.position()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (bool(event.buttons()) & bool(Qt.LeftButton)) & self.drawing:

            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(
                QPen(self.brushColor, self.brushSize, Qt.SolidLine,
                     Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.position())

            # change the last point
            self.lastPoint = event.position()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.modified = True
        self.update()

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if filePath == "":
            return
        self.image.save(filePath)

    def QImageToCvMat(self):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = self.image.convertToFormat(QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr


class MyForm(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MyForm, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.scribbleArea = ScribbleArea()
        self.scribbleArea.setParent(self.ui.frame_scribble)

        # Run Button Clicked
        self.ui.pushButton_run.clicked.connect(self.run)
        # Clear Button Clicked
        self.ui.pushButton_clean.clicked.connect(self.clean)
        # Translate Button Clicked
        self.ui.pushButton_translate.clicked.connect(self.translate)

        # init the Text of the label
        self.ui.label_prediction.setText("Prediction ID:")
        self.ui.label_english.setText("English:")
        self.ui.label_chinese.setText("Chinese 中文: ")

        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b2 = nn.Sequential(*resnet_block(64, 64, 256, 3, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(256, 128, 512, 4, 2))
        b4 = nn.Sequential(*resnet_block(512, 256, 1024, 6, 2))
        b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 2, 2))
        self.model = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(2048, 1588))

        checkpoint = torch.load('./checkpoint_ep0600.pth',  map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.x = 0
        self.show()

    def translate(self):
        # TO-DO Chinese to English Translation
        return

    def clean(self):
        self.ui.label_prediction.setText("Prediction ID:")
        self.ui.label_english.setText("English:")
        self.ui.label_chinese.setText("Chinese 中文: ")
        self.scribbleArea.clearImage()
        self.ui.label_chinese.setStyleSheet(
            "background-color: lightgrey; border: 1px solid black;")
        self.ui.lineEdit_chinese.setText("")


    def ai_predict(self, image_path):
        # 利用临时存照片来实现关联
        image_path = image_path[:, :, 0]
        im = Image.fromarray(image_path)
        im.save("2.png")

        test_dataset = TestData()
        test_loader = DataLoader(test_dataset, shuffle=True,  batch_size = 1,pin_memory=True)
        prediction_top_10 = None
        with torch.no_grad():
            for image, label,path in test_loader:
                prediction_top_10 = self.model(image)
                sorted_tensor, indices = torch.sort(prediction_top_10[0])
        return prediction_top_10


    def run(self):
        self.x = self.x + 1
        print("Run Button Clicked {}".format(self.x))

        image_np = self.scribbleArea.QImageToCvMat()

        pred = self.ai_predict(image_np)
        # 可以加可以不加
        pred = torch.sigmoid(pred)
        prediction_top_10 = pred

        prediction_top_10 = prediction_top_10[0]
        print(prediction_top_10, prediction_top_10.shape)
        pred_label = torch.argmax(prediction_top_10)
        pred_prob = prediction_top_10[pred_label]

        with open('Validation_label.json', 'r', encoding='utf8') as f:
            labels = json.load(f)
            id_name = {}
            for k in labels:
                id_name[labels[k]] = k
            
        with open('ID_to_chinese.json', 'r', encoding='utf8') as f:
            ids = json.load(f)

        # 获取第一
        c_name = id_name[pred_label.item()]
        if "_" in c_name:
             c_name = c_name.split("_")[0]
        final_c = ids[c_name]

        # 获取前5
        top_k = 5
        top3_values, top3_indices = torch.topk(prediction_top_10, top_k)
        top_k_character = []
        top_k_prob = []
        for i in range(top_k):
            top_k_prob.append([top3_values[i].item()])
            c_name_t = id_name[top3_indices[i].item()]

            if "_" in c_name_t:
                c_name_t = c_name_t.split("_")[0]
            final_c_t = ids[c_name_t]
            top_k_character.append(final_c_t)
            top_k_character.append(c_name_t)
        self.ui.lineEdit_chinese.setText(' '.join(top_k_character))

        print("label:", pred_label, " prob:", pred_prob)
        print("topk prob:", top_k_prob)
        pred_character = final_c
        self.ui.label_prediction.setText(
            f"Prediction ID: {pred_label} \nAcc: {pred_prob:.8f}")
        self.ui.label_chinese.setText("Chinese 中文: " + pred_character)
        if pred_prob > 0.5:
            self.ui.label_chinese.setStyleSheet(
                "background-color: lightgreen; border: 1px solid black;")

        elif pred_prob < 0.5 and pred_prob > 0.0001:
            self.ui.label_chinese.setStyleSheet(
                "background-color: lightyellow; border: 1px solid black;")
        else:
            self.ui.label_chinese.setStyleSheet(
                "background-color: red; border: 1px solid black;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec())
