# -*- coding: utf-8 -*-
# 최신 PyQt5 환경 대응:
#  - matplotlib 백엔드 사전 지정(plt 이전)
#  - PyQt5를 cv2보다 먼저 import (Qt 플러그인 경로 꼬임 방지)
#  - PyQt5의 Qt/Qt5 plugins 경로 자동 추가(qwindows.dll 탐색 보조)
#  - QDesktopWidget 대체(권장)로 화면 중앙 배치
#  - 클릭 좌표 None 방어, 메시지 안내 수정(왼쪽=시작, 오른쪽=끝)
#  - 저장 시 dtype 보정, 히스토그램 컬러/그레이 자동 처리

import sys
import os
import numpy as np
import subprocess

# ❶ Matplotlib 백엔드 지정은 pyplot import 전에!
import matplotlib
matplotlib.use("Qt5Agg")

# ❷ PyQt5 먼저 import
import PyQt5
from PyQt5.QtCore import QCoreApplication, Qt, QDate, QTime
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog,
    QMessageBox, QAction
)

# Qt 플러그인 경로를 자동으로 추가(폴더명이 Qt 또는 Qt5 모두 지원)
def _add_qt_plugin_paths():
    base = os.path.dirname(PyQt5.__file__)
    candidates = [
        os.path.join(base, "Qt", "plugins"),
        os.path.join(base, "Qt5", "plugins"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            QCoreApplication.addLibraryPath(p)
            # 환경변수도 보조로 지정(디버거/런처에 따라 필요할 수 있음)
            os.environ.setdefault("QT_PLUGIN_PATH", p)
_add_qt_plugin_paths()

# ❸ 마지막에 cv2 import
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Initialize
        self.image = None
        self.startX = None
        self.startY = None
        self.EndX = None
        self.EndY = None
        self.UseROI = False
        self.UseCropImage = False
        self.fname = ""
        # 아이콘 경로 설정 (소스 기준 ICON 폴더)
        ICON_DIR = os.path.join(os.path.dirname(__file__), "ICON")

        # Title
        self.setWindowTitle('ImageViewer')
        self.setWindowIcon(QIcon(os.path.join(ICON_DIR, 'Icon.png')))

        # Toolbar Actions
        QAopenImage = QAction(QIcon(os.path.join(ICON_DIR, 'open.png')), 'Open the Image', self)
        QAopenImage.setShortcut(QKeySequence.Open)
        QAopenImage.triggered.connect(self.openImage)

        QAshowHistogram = QAction(QIcon(os.path.join(ICON_DIR, 'hist.png')), 'Show the Histogram', self)
        QAshowHistogram.triggered.connect(self.showHistogram)

        QAsaveImage = QAction(QIcon(os.path.join(ICON_DIR, 'save.png')), 'Save the Image', self)
        QAsaveImage.triggered.connect(self.saveImage)

        QAsaveSlideShow = QAction(QIcon(os.path.join(ICON_DIR, 'slideshow.png')), 'Slide Show', self)
        QAsaveSlideShow.triggered.connect(self.slideShow)

        QAOpenNotePad = QAction(QIcon(os.path.join(ICON_DIR, 'notepad.png')), 'Open the NotePad', self)
        QAOpenNotePad.triggered.connect(self.openNotePad)

        QAGetROI = QAction(QIcon(os.path.join(ICON_DIR, 'ROI.png')), 'Get the ROI', self)
        QAGetROI.triggered.connect(self.getROI)

        QAShowCropImage = QAction(QIcon(os.path.join(ICON_DIR, 'cropopen.png')), 'Show the CropImage', self)
        QAShowCropImage.triggered.connect(self.showCropImage)

        QAShowNormImage = QAction(QIcon(os.path.join(ICON_DIR, 'normalize.png')), 'Show the Normalize Image', self)
        QAShowNormImage.triggered.connect(self.showNormImage)

        QAShowEqualImage = QAction(QIcon(os.path.join(ICON_DIR, 'eqaulize.png')), 'Show the Equalize Image', self)
        QAShowEqualImage.triggered.connect(self.showEqualImage)


        # Toolbars
        tb1 = self.addToolBar('Actions')
        tb1.addAction(QAopenImage)
        tb1.addAction(QAshowHistogram)
        tb1.addAction(QAsaveImage)
        tb1.addAction(QAsaveSlideShow)
        tb1.addAction(QAOpenNotePad)

        tb2 = self.addToolBar('Actions2')
        tb2.addAction(QAGetROI)
        tb2.addAction(QAShowCropImage)

        tb3 = self.addToolBar('Actions3')
        tb3.addAction(QAShowNormImage)
        tb3.addAction(QAShowEqualImage)

        # Central (Matplotlib Canvas + Toolbar)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_press)

        vbox = QVBoxLayout(self.main_widget)
        vbox.addWidget(self.canvas)
        self.addToolBar(Qt.BottomToolBarArea, NavigationToolbar(self.canvas, self))

        # Window Location
        self.resize(1200, 800)
        self.center()
        self.show()

    def openNotePad(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Text File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if fileName:
            subprocess.Popen(['notepad.exe', fileName])

    def center(self):
        # 권장 방식: 현재 스크린 기준 중앙 배치
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.center() - self.rect().center())

    def getImagePath(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 'Images (*.bmp *.jpg *.png);;All Files (*)', options=options
        )
        self.fname = fname
        return fname

    def getImagesPath(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fnames, _ = QFileDialog.getOpenFileNames(
            self, 'Open Images', '', 'Images (*.bmp *.jpg *.png);;All Files (*)', options=options
        )
        # 다중 선택 시 self.fname은 리스트가 되므로, 단일 경로가 필요한 함수에서는 주의
        self.fname = fnames
        return fnames

    def on_press(self, event):
        # 캔버스 밖을 클릭하거나 좌표가 None인 경우 방어
        if event.xdata is None or event.ydata is None:
            return

        if self.UseROI:
            if event.button == 1:  # Left = Start
                self.startX, self.startY = int(event.xdata), int(event.ydata)
                print(f"Start Coordinates: ({self.startX}, {self.startY})")
                self.Information_event("Start Position Done ! (왼쪽 클릭)")
            elif event.button == 3:  # Right = End
                self.UseROI = False
                self.EndX, self.EndY = int(event.xdata), int(event.ydata)
                print(f"End Coordinates: ({self.EndX}, {self.EndY})")
                self.Information_event("End Position Done ! (오른쪽 클릭)")

                # 영역 표시
                self.figure.clf()
                image_axes = self.figure.add_subplot(111)
                image_axes.imshow(self.image, cmap=plt.cm.gray)

                rect = patches.Rectangle(
                    (self.startX, self.startY),
                    (self.EndX - self.startX),
                    (self.EndY - self.startY),
                    linewidth=2,
                    edgecolor='cyan',
                    fill=False
                )
                image_axes.add_patch(rect)
                self.canvas.draw()
                self.UseCropImage = True

    def getROI(self):
        if self.image is None:
            self.Information_event("이미지가 없습니다. [Open the Image]로 이미지를 등록하세요.")
            return
        self.UseROI = True
        self.Information_event("ROI 지정: 시작=왼쪽 클릭, 종료=오른쪽 클릭")

    def showCropImage(self):
        if self.UseCropImage:
            y0, y1 = self.startY, self.EndY
            x0, x1 = self.startX, self.EndX
            if None in (x0, y0, x1, y1):
                self.Information_event("ROI가 올바르지 않습니다. 다시 지정하세요.")
                return
            h, w = self.image.shape[:2]
            y0, y1 = max(0, y0), min(h, y1)
            x0, x1 = max(0, x0), min(w, x1)
            if y1 <= y0 or x1 <= x0:
                self.Information_event("ROI 영역이 잘못되었습니다.")
                return

            crop = self.image[y0:y1, x0:x1].copy()
            self.image = crop
            self.showImage()
            self.UseCropImage = False
        else:
            self.Information_event("[Get the ROI] 버튼으로 ROI를 먼저 설정하세요.")

    def showNormImage(self):
        if self.image is None:
            self.Information_event("이미지가 없습니다. [Open the Image]로 이미지를 등록하세요.")
            return

        self.Information_event("Normalization: 0~255 전구간으로 선형 스케일링")
        tm = cv2.TickMeter(); tm.reset(); tm.start()

        # OpenCV normalize는 dtype 보정 없이도 동작하지만, 표시를 위해 uint8로 캐스팅
        norm = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
        if norm.dtype != np.uint8:
            norm = norm.astype(np.uint8)

        tm.stop()
        print('Normalize Processing Time: {}ms.'.format(tm.getTimeMilli()))

        self.image = norm
        self.showImage()

    def showEqualImage(self):
        if self.image is None:
            self.Information_event("이미지가 없습니다. [Open the Image]로 이미지를 등록하세요.")
            return

        self.Information_event("Equalization: 히스토그램 평활화")
        tm = cv2.TickMeter(); tm.reset(); tm.start()

        # 원본 파일 경로에서 그레이로 다시 읽기(ROI 적용을 위해)
        src_path = self.fname if isinstance(self.fname, str) else (self.fname[0] if self.fname else "")
        if not src_path or not os.path.isfile(src_path):
            self.Information_event("원본 파일 경로를 찾을 수 없습니다. 이미지를 다시 여세요.")
            return

        dst = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if dst is None:
            self.Information_event("이미지 읽기 실패.")
            return

        if self.startX is None or self.startY is None or self.EndX is None or self.EndY is None:
            crop = dst
        else:
            y0, y1 = sorted([self.startY, self.EndY])
            x0, x1 = sorted([self.startX, self.EndX])
            crop = dst[y0:y1, x0:x1]

        rst = cv2.equalizeHist(crop)

        tm.stop()
        print('Equalize Processing Time: {}ms.'.format(tm.getTimeMilli()))

        self.image = rst
        self.showImage()

    def openImage(self):
        fname = self.getImagePath()
        if not fname:
            return
        try:
            # plt.imread는 dtype이 float일 수 있어 후처리에서 주의 필요
            img = plt.imread(fname)
            # 알파 채널 제거
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            self.image = img
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file\n{e}", QMessageBox.Ok)
            return
        self.showImage()

    def Information_event(self, text):
        QMessageBox.information(self, 'Information', text)

    def saveImage(self):
        if self.image is None:
            self.Information_event("이미지가 없습니다. [Open the Image]로 이미지를 등록하세요.")
            return

        # 저장 시 dtype 보정
        img = self.image
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # RGB를 BGR로 변환하여 저장(plt.imread로 읽은 경우 대비)
        if img.ndim == 3 and img.shape[2] == 3:
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img

        DateNow = QDate.currentDate()
        TimeNow = QTime.currentTime()
        out_name = f"{DateNow.toString('yyyy.MM.dd')}.{TimeNow.toString('hh.mm.ss')}.bmp"
        cv2.imwrite(out_name, img_to_save)
        self.Information_event(f"Save Image Complete !!\n{out_name}")

    def showImage(self):
        if self.image is None:
            return
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        # 그레이/컬러 자동 처리
        if self.image.ndim == 2:
            ax.imshow(self.image, cmap=plt.cm.gray)
        else:
            ax.imshow(self.image)
        ax.axis('off')
        self.canvas.draw()

    def slideShow(self):
        img_files = self.getImagesPath()
        if not img_files:
            self.Information_event("선택한 이미지가 없습니다.")
            return

        cnt = len(img_files)
        idx = 0
        while True:
            img = plt.imread(img_files[idx])
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            self.image = img
            self.showImage()

            # 1초 간격, 키 입력 시 종료
            if cv2.waitKey(1000) >= 0:
                break

            idx += 1
            if idx >= cnt:
                break

        self.Information_event("Slide Show Done!!")

    def showHistogram(self):
        if self.image is None:
            self.Information_event("이미지가 없습니다. [Open the Image]로 이미지를 등록하세요.")
            return

        dst = self.image.copy()
        self.figure.clf()

        tm = cv2.TickMeter(); tm.reset(); tm.start()

        ax = self.figure.add_subplot(111)

        # 그레이/컬러 자동 판별
        if dst.ndim == 2:
            hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
            ax.plot(hist)
            ax.set_title("Grayscale Histogram")
        else:
            # RGB 각 채널 히스토그램
            channels = cv2.split(cv2.cvtColor((dst*255).astype(np.uint8) if dst.dtype!=np.uint8 else dst, cv2.COLOR_RGB2BGR))
            labels = ['B', 'G', 'R']
            for ch, lab in zip(channels, labels):
                h = cv2.calcHist([ch], [0], None, [256], [0, 256])
                ax.plot(h, label=lab)
            ax.legend()
            ax.set_title("Color Histogram (BGR)")

        tm.stop()
        print('Histogram Processing Time: {}ms.'.format(tm.getTimeMilli()))

        self.canvas.draw()

    @staticmethod
    def getGrayHistImage(hist):
        imgHist = np.full((100, 256), 255, dtype=np.uint8)
        histMax = np.max(hist)
        if histMax == 0:
            return imgHist
        for x in range(256):
            pt1 = (x, 100)
            pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
            cv2.line(imgHist, pt1, pt2, 0)
        return imgHist


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    # PyQt5는 전통적으로 exec_였지만, 최신에선 exec도 가능
    if hasattr(app, "exec"):
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())
