# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

import kivy

from kivy_deps import sdl2, glew
import tensorflow as tf
import cv2
import os, sys
import numpy as np
import time 
import random 
import math 
import datetime
import joblib
import pyaudio




a = Analysis(
    ['..\\app.py'],
    pathex=[],
    binaries=[],
    datas=[
	("..\\resources/app.kv", "resources"),
	("..\\resources/std_scaler.bin", "resources"),
    ("..\\images/camera_overlay.png", "images"),
	("..\\resources/haarcascade_frontalface_default.xml", "resources"),
	("..\\resources/positive_negative_classifier.h5", "resources"),
	("..\\resources/speech_classifier.h5", "resources"),
    ("..\\resources/speech_classifier.h5", "resources"),
    ("..\\kivy_venv/Lib/site-packages/librosa/util/example_data/registry.txt", "librosa/util/example_data"),
    ("..\\kivy_venv/Lib/site-packages/librosa/util/example_data/index.json", "librosa/util/example_data")

    ],

    hiddenimports=["pyaudio", "sklearn.utils._typedefs","sklearn.neighbors._ball_tree","sklearn.neighbors._ball_tree", "sklearn.neighbors._partition_nodes"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='test',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
	*[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
