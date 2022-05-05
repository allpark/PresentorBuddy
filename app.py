# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, ClearColor, ClearBuffers, Fbo, Line, Rectangle
from kivy.logger import Logger
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.resources import resource_add_path, resource_find

# Import other dependencies
import cv2
import tensorflow as tf
import os, sys
import numpy as np
import time 
import random 
import math 
import datetime

# emotion model testing
from joblib import load
import librosa

import threading
import subprocess
import pyaudio 
import wave 

# path declaration 
VIRTUAL_PATH_MODE = hasattr(sys, '_MEIPASS') 

# define application file paths
PATHS = {
    "face_classifier" :  "resources/positive_negative_classifier.h5",
    "haar" : "resources/haarcascade_frontalface_default.xml",
    "camera_overlay" : "resources/cam_overlay.png",
    "scaler" : "resources/std_scaler.bin",
    "speech_classifier" : "resources/speech_classifier.h5",
    "temp_audio_path" : "temp"
}

# if application has been frozen as exe then use virtual paths for files
if (VIRTUAL_PATH_MODE):
    for key, path in PATHS.items():
        PATHS[key] = sys._MEIPASS + "/" + path 

# speech classifier helper 
class speech_classifier():
    def __init__(self, classifier_path=PATHS["speech_classifier"], data_scaler_path=PATHS["scaler"]):
        self.classifier = tf.keras.models.load_model(classifier_path)
        self.data_scaler = load(data_scaler_path)

    def extract_features(self, data, sample_rate):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) 
        
        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) 

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) 
        return result

    def get_features(self, path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)
        return result

    def analyse_emotion(self, wav_path):
        if not(os.path.isfile(wav_path)):
            raise ValueError("error loading " + wav_path)

        # extract audio features 
        audio_features = np.asarray([self.get_features(wav_path)])

        # scale audio features using the original scaler fitted to training data
        scaled_audio = self.data_scaler.transform(audio_features)

        # return probabilities  
        return self.get_sentiment_and_emotion(self.classifier.predict(scaled_audio)[0])

    def get_sentiment_and_emotion(self, class_probs):
        emotion_extractor = {0:"angry", 1:"calm", 2:"disgust", 3:"fear", 4:"happy", 5:"neutral", 6:"sad", 7:"surpise"}
        positivity_sentiment = class_probs[1] + class_probs[4] + class_probs[5] + class_probs[7]  
        negativity_sentiment = class_probs[0] + class_probs[2] + class_probs[3] + class_probs[6]
        emotion_detected = emotion_extractor[np.argmax(class_probs)]
        return {
            "emotion": emotion_detected, 
            "emotion_confidence": class_probs.max(axis=-1),
            "positivity_sentiment" : positivity_sentiment,
             "negativity_sentiment" : negativity_sentiment
        }

class audio_stream_recorder():
    def __init__(self, save_as, recorder_indexer):
        # file output vars
        self.save_as = save_as
        # recorder state variables
        self.isRecording = True
        self.pyaudio = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=recorder_indexer,
            frames_per_buffer = 2**12
        )
        self.audio_bytes = []

    # method for recording audio stream using pyaudio by reading audio source frames
    def record(self):
        self.audio_stream.start_stream()
        while self.isRecording:
            try:
                self.audio_bytes.append(self.audio_stream.read(2**12))
            except Exception as error:
                print(error)
            time.sleep(0.01)

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pyaudio.terminate()

        waveFile = wave.open(self.save_as, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(self.audio_bytes))
        waveFile.close()

    def start(self):
        threading.Thread(target=self.record).start()
    def stop(self):
        if self.isRecording: self.isRecording = False


# declare both screens
class Main(Screen):
    def __init__(self, **kwargs): 
        super(Main, self).__init__(**kwargs)
        # states 
        self.startTime   = time.time()  
        self.endTime     = time.time()

        self.isRecording = False 
        self.faceDetected = False 
        self.audioWritten = False 
        self.lastPositivity = 0 
        self.positivityArray = []
        self.detectedFaceAccumulator = 0

        # audio recorder dump states
        self.queueInitialised = False 
        self.queueResolved = False 
        self.audio_temp_file_name = "temp_audio.wav"
        self.audio_metadata = {}

        # inputs and camera 
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cameraPreview =  Texture.create(size=(640, 480), colorfmt='bgra')
        self.cameraExtractedFace =  Texture.create(size=(48, 48), colorfmt='bgr')
        self.cameraResolution = (640, 480)

        # Load pre-trained models and their respective components 
        self.model = tf.keras.models.load_model(PATHS["face_classifier"])
        self.speech_classifier = speech_classifier()
        self.audio_recorder    = audio_stream_recorder(save_as = self.audio_temp_file_name, recorder_indexer=0)
        self.face_detector     = cv2.CascadeClassifier(PATHS["haar"])

        with self.canvas:
            self.fbo  = Fbo(size=self.cameraResolution)
    
        with self.fbo:

            cosmeticColor = self.getPositivityAsColor()

            ClearColor(0, 0, 0, 1)
            ClearBuffers()
            
            self.cameraOverlay            = Rectangle(source = PATHS["camera_overlay"], size=self.cameraResolution)
            self.cameraOverlayIcon        = Rectangle(source = PATHS["camera_overlay"], size=self.cameraResolution)

            Color(0,1,0,1)
            self.detectedFaceBoundingBox  = Line(width=2)
            self.detectedFaceLabel        = Label(text="Face detected", color=(0,1,0,1), pos=(0, 0), valign='top', halign='left', text_size=(200, 50))
            self.detectedFaceEmotionLabel = Label(text="Emotion: positive", color=(0,1,0,1), pos=(0, 0), valign='top', halign='left', text_size=(200, 50))

            self.elapsedTimeLabel = Label(text="Elapsed time: ", color=(0,1,0,1), pos=(0, 0), valign='top', halign='center', text_size=(200, 50))

        Clock.schedule_interval(self.update, 1.0/20.0)

    # function for waiting for audio to be saved from stream
    def queueSetupPageForProcessing(self, *args):
        # check if queuer timer has been initialised
        if (not(self.queueInitialised)):
            Clock.schedule_interval(self.queueSetupPageForProcessing, 1.0)
            self.queueInitialised = True 
            self.queueResolved = False 

        #                exit if statement 
        # check if queue has been initialsed and resolved
        if (self.queueInitialised and self.queueResolved):
            Clock.unschedule(self.queueSetupPageForProcessing)
            self.queueInitialised = False 

            # compute speech emotion metadata
            print("converting")
            self.audio_metadata = self.speech_classifier.analyse_emotion(self.audio_temp_file_name)
            print("FINISHED")
            # switch to analysis page 
            self.setupAnalysisPage()
            print("finished displaying")

        # check if queue can be now resolved (if audio file exists)
        if (not(self.queueResolved)):
            if os.path.isfile(self.audio_temp_file_name):
                self.queueResolved = True 
        # repeat 

    # function for updating camera id type
    def camera_selector(self, id):
        # update camera input 
        self.capture = cv2.VideoCapture(int(id))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # called when record is pressed 
    # used for application state change and processing
    def on_record_press(self):
        self.isRecording = not(self.isRecording)
        # if video is being recorded
        if (self.isRecording):
            self.startTime = time.time() 
            self.positivityArray = []
            self.detectedFaceAccumulator = 0
            self.ids.btn_mainscreen_record.text = "Stop Recording"
            self.audio_recorder    = audio_stream_recorder(save_as = self.audio_temp_file_name, recorder_indexer=0)
            self.audio_recorder.start()
            self.audioWritten = False 
        # if video has finished recording
        else:
            self.endTime = time.time() 
            self.ids.btn_mainscreen_record.text = "Start Recording"
            self.audio_recorder.stop()
            self.queueSetupPageForProcessing()
            #self.cameraOverlay.source = PATHS["camera_overlay"]

    # function for getting visual score
    def getVisualScore(self):
        if len(self.positivityArray)==0:
            return 0
        else:
            return (sum(self.positivityArray)/len(self.positivityArray))*100

    def getSpeechScore(self):
        return round(self.audio_metadata["positivity_sentiment"] * 100)
    def getSpeechEmotionConfidence(self):
        return round(self.audio_metadata["emotion_confidence"] * 100)
    def getSpeechEmotion(self):
        return self.audio_metadata["emotion"]

    # functions for converting sentiment scores to feedback
    def getAudioFeedbackAsString(self, score, emotion, confidence):
        if score >= 0 and score < 20:
            return "Emotion ("+emotion+") detected, confidence: " + confidence + ". Needs great improvement."
        elif score >= 21 and score < 40:
            return "Emotion ("+emotion+") detected, confidence: " + confidence + ". Needs improvement." 
        elif score >= 41 and score < 60:
            return"Emotion ("+emotion+") detected, confidence: " + confidence + ". Needs slight improvement." 
        else:
            return "Emotion ("+emotion+") detected, confidence: " + confidence + ". Needs no improvement."
    def getVisualFeedbackAsString(self, score):
        if score >= 0 and score < 20:
            return "Overwhelmingly negative emotion detected. Needs great improvement."
        elif score >= 21 and score < 40:
            return "Negative emotion detected. Needs improvement."
        elif score >= 41 and score < 60:
            return"Neutral emotion detected. Needs slight improvement."
        else:
            return "Positive emotion detected. Needs no improvement."

    # setup presentation analysis page
    def setupAnalysisPage(self):

        self.parent.current = 'video_analysis'
        analysisPageContext = self.manager.get_screen("video_analysis")

        label_overall_score =  analysisPageContext.ids.label_overall_score
        label_visual_score =  analysisPageContext.ids.label_visual_score
        label_audio_score =  analysisPageContext.ids.label_audio_score

        # compute speech positivity score 
        speechPositivity = self.getSpeechScore()
        speechEmotion    = self.getSpeechEmotion()
        speechEmotionConfidence = str(self.getSpeechEmotionConfidence()) + "%"

        # compute image positivity score 
        visualPositivity = self.getVisualScore()
        totalPositivity  = visualPositivity * 0.6 + speechPositivity * 0.4 

        label_visual_score.text = "Visual: " + str(round(visualPositivity)) + "/100"
        label_overall_score.text = "Overall Score: " + str(round(totalPositivity)) + "/100"
        label_audio_score.text = "Audio: " + str(round(speechPositivity)) + "/100"
    
        # generating feedback here
        label_feedback_visual =  analysisPageContext.ids.label_feedback_visual
        label_feedback_audio =  analysisPageContext.ids.label_feedback_audio


        # generating facial feedback here
        if self.detectedFaceAccumulator < 10:
            label_feedback_visual.text = "Visual: face not detected"
        else:
            label_feedback_visual.text = "Visual: " + self.getVisualFeedbackAsString(visualPositivity)

        # generating audio feedback here
        label_feedback_audio.text = "Audio: " + self.getAudioFeedbackAsString(speechPositivity, speechEmotion.capitalize(), speechEmotionConfidence )
        

    # methods for extracting facial emotion from a frame 
    def classify_emotion(self, frames):
        pre_processed_frames = np.asarray(frames, dtype=np.float32) / 255.0
        return self.model.predict(pre_processed_frames)
    def detect_face(self, frame):
        faces = self.face_detector.detectMultiScale(frame, 1.1, 6)
        if (faces==()):
            return False, (), ()
        else:
            x, y, w, h = faces[0]
            return True, (x, y), (x+w, y+h)
    def extract_face(self, img, bb0, bb1, width=48, height=48):
        return cv2.resize( img[bb0[1]:bb1[1], bb0[0]:bb1[0]],  (width, height), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    
    # redraw and update face bounding box and its label properties
    def update_face_boundingbox(self, x, y, x2, y2):
        self.detectedFaceBoundingBox.rectangle = (x, self.cameraResolution[1] - y, x2-x,   y-y2)
        self.cameraOverlay.texture  = self.cameraPreview 
        self.detectedFaceLabel.pos = (int((x+x2)*0.5), int(self.cameraResolution[1] - y) - 50 )
        self.detectedFaceEmotionLabel.pos = (int((x+x2)*0.5), int(self.cameraResolution[1] - y - (y2-y)) - 80 )
    # main loop responsible for retrieving face emotion in real time
    def update(self, *args):
        if not(self.isRecording): return

        # Read frame from opencv
        ret, frame = self.capture.read()
        
        # error check for null frames 
        if ret is None or frame is None: return 

        
        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        detected, bb0, bb1 = self.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if (detected):
            extracted_face = self.extract_face(frame, bb0, bb1)    
            self.update_face_boundingbox(bb0[0], bb0[1], bb1[0], bb1[1])
            grey   = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
            frames = [grey] 
            self.lastPositivity = self.classify_emotion([grey])[0][1]
            self.positivityArray.append(self.lastPositivity)


            self.detectedFaceAccumulator += 1
        else:
            self.update_face_boundingbox(0,0,0,0)

        self.updateCosmetics(detected)
        self.cameraPreview = img_texture
        self.ids.webcam_preview.texture = self.fbo.texture
        self.faceDetected = detected 
        self.elapsedTimeLabel.text = str(datetime.timedelta(seconds=round(time.time() - self.startTime)))

    # update face bounding box labels and cosmetics
    def updateCosmetics(self, state):
        if (state):
            self.detectedFaceLabel.text = "Face detected"
            self.detectedFaceEmotionLabel.text = "Emotion: " + self.getPositivityAsCategory()
            self.detectedFaceEmotionLabel.color = self.getPositivityAsColor()
        else:
            self.detectedFaceLabel.text = ""
            self.detectedFaceEmotionLabel.text = ""

    def getPositivityAsCategory(self):
        if self.lastPositivity > 0.6:
            return "positive"
        elif (self.lastPositivity > 0.3 and self.lastPositivity <= 0.6):
            return "neutral"
        else:
            return "negative"

    def getPositivityAsColor(self):
        if self.lastPositivity > 0.7:
            return (0,1,0,1)
        elif (self.lastPositivity > 0.3 and self.lastPositivity <= 0.7):
            return (0.5, 0.5, 0.5, 1)
        else:
            return (1,0,0,1)

class VideoAnalysisScreen(Screen):
    pass

class TestApp(App):
    isRecording = False 
    def on_record_press(self, *args):
        isRecording = not(isRecording)
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Main(name='main'))
        sm.add_widget(VideoAnalysisScreen(name='video_analysis'))
        self.title = 'Presenter Buddy'

        return sm
if __name__ == '__main__':
    if hasattr(sys, '_MEIPASS'):
        resource_add_path(os.path.join(sys._MEIPASS))

    Builder.load_file('resources/app.kv')
    TestApp().run()


    