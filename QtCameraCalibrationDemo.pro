QT += core gui widgets

TARGET = CameraCalibration
TEMPLATE = app
CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    camera_calibration.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    out_camera_data.yml \
    out_camera_data.xml \
    config.xml \
    config_chess.xml \
    config_circle.xml \
    config_ring.xml \
    out_camera_data.xml

OPENCV_DIR = "D:\opt\opencv\build"

INCLUDEPATH += $$OPENCV_DIR/include
QMAKE_LIBDIR += $$OPENCV_DIR/x64/vc15/lib

CONFIG(debug, debug|release) {
    LIBS += -lopencv_world344d
}
else {
    LIBS += -lopencv_world344
}

HEADERS += \
    metrics.h \
    geometria.h
