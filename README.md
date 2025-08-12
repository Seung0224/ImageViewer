# ImageViewer

Windows 환경에서 **PyQt5**와 **Matplotlib**를 이용해  
이미지 파일을 열고, ROI 지정, 히스토그램 분석, 정규화/히스토그램 평활화, 슬라이드쇼, 크롭, 저장 등을 수행할 수 있는 Python 기반 이미지 뷰어입니다.  

---

<img width="1202" height="832" alt="image" src="https://github.com/user-attachments/assets/f478b566-2a38-4f85-9d2a-adde505ef1ce" />

## 📦 프로젝트 개요

- **플랫폼:** Windows 10 / Python 3.x  
- **UI 프레임워크:** PyQt5  
- **이미지 처리:** OpenCV, Matplotlib  
- **목적:** 단일 이미지 또는 다중 이미지를 로드하고 다양한 시각화·분석 기능 제공  

---

## ✅ 주요 기능

### 1. 📂 이미지 로드 & 저장
- 단일 이미지 파일 열기 (`.bmp`, `.jpg`, `.png`)
- 다중 이미지 로드 후 슬라이드쇼 재생
- 날짜·시간 기반 파일명으로 이미지 저장

### 2. 🔍 ROI(Region of Interest) 지정
- **왼쪽 클릭:** 시작 좌표 지정
- **오른쪽 클릭:** 종료 좌표 지정
- 지정 영역 표시 및 Crop 기능 제공

### 3. 🎨 이미지 처리
- **정규화(Normalization):** 픽셀 값을 0~255 범위로 스케일 조정
- **히스토그램 평활화(Equalization):** 대비 향상 처리
- **히스토그램 표시:** 그레이스케일 히스토그램 시각화

### 4. 🖼️ 이미지 탐색
- 슬라이드쇼로 다중 이미지 순차 표시
- Matplotlib Navigation Toolbar로 확대/이동 지원

### 5. 🗒️ 기타 기능
- 메모장 실행 및 텍스트 파일 열기
- ROI 적용 후 크롭 이미지 보기
- 사용자 정보 메시지 팝업 제공

---

## 🧰 사용 방법

1. Python 환경에 필요한 라이브러리 설치  
   ```bash
   pip install pyqt5 matplotlib opencv-python numpy
