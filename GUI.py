import os
import sys
import io
import cv2
import numpy as np
from PIL import Image as PILImage, ImageOps, UnidentifiedImageError
import time

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

import mediapipe as mp

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle


if sys.platform not in ('android', 'ios'):
    Window.size = (700, 650)
    Window.clearcolor = (0.95, 0.95, 0.98, 1)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path("model/Gender_prediction_final.h5")
THRESHOLD_PATH = resource_path("model/best_threshold.txt")
IMG_SIZE = 224
PREDICTION_FREQUENCY = 10

class LoadingScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)

        with self.canvas.before:
            Color(0.88, 0.88, 0.92, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.label_top = Label(
            text="Loading AI model...\nPlease wait a moment",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_top.bind(size=self.label_top.setter('text_size'))
        self.add_widget(self.label_top)

        self.loading_image = KivyImage(
            source=resource_path('assets/LOGO.png'),
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.6)
        )
        self.add_widget(self.loading_image)

        self.label_bottom = Label(
            text="Powered by Leonardo Cofone",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_bottom.bind(size=self.label_bottom.setter('text_size'))
        self.add_widget(self.label_bottom)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class GenderApp(App):
    def build(self):
        self.title = "Gender Prediction - By Leonardo Cofone"
        self.model = None
        self.best_threshold = 0.5
        self.cap = None
        self.live_mode = False
        self.frame_count = 0
        self.last_label = None
        self.last_box = None
        
        self.last_prediction_time = 0


        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.root_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.loading_screen = LoadingScreen()
        self.root_layout.add_widget(self.loading_screen)

        Clock.schedule_once(self.load_model, 1)

        return self.root_layout

    def load_model(self, dt):
        try:
            self.model = load_model(MODEL_PATH)
            with open(THRESHOLD_PATH, 'r') as f:
                self.best_threshold = float(f.read())
            self.show_main_screen()
        except Exception as e:
            self.loading_screen.label.text = f"Failed to load model:\n{e}"

    def show_main_screen(self):
        self.root_layout.clear_widgets()
        Window.clearcolor = (0.88, 0.88, 0.92, 1)

        self.title_label = Label(
            text="Gender Prediction AI",
            font_size='28sp',
            size_hint=(1, 0.1),
            color=(0.1, 0.1, 0.1, 1),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        self.root_layout.add_widget(self.title_label)

        self.image_panel = KivyImage(
            size_hint=(1, 0.65),
            allow_stretch=True,
            keep_ratio=True,
            source=""
        )
        self.root_layout.add_widget(self.image_panel)

        self.result_label = Label(
            text="Upload an image or start live camera",
            font_size='20sp',
            size_hint=(1, 0.1),
            color=(0.1, 0.1, 0.1, 1),
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        self.root_layout.add_widget(self.result_label)

        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=15)

        btn_color_pink = (0.8, 0.3, 0.5, 1)
        btn_text_color = (1, 1, 1, 1)

        self.upload_btn = Button(text="Upload Image", background_color=btn_color_pink, color=btn_text_color)
        self.upload_btn.bind(on_release=self.open_file_chooser)
        btn_layout.add_widget(self.upload_btn)

        self.live_btn = Button(text="Start Live Cam", background_color=btn_color_pink, color=btn_text_color)
        self.live_btn.bind(on_release=self.toggle_live_cam)
        btn_layout.add_widget(self.live_btn)

        self.reset_btn = Button(text="Reset", background_color=btn_color_pink, color=btn_text_color)
        self.reset_btn.bind(on_release=self.reset_app)
        btn_layout.add_widget(self.reset_btn)

        self.root_layout.add_widget(btn_layout)

    def open_file_chooser(self, instance):
        if self.live_mode:
            self.stop_live_cam()

        home = os.path.expanduser("~")
        downloads = os.path.join(home, 'Downloads')
        pictures = os.path.join(home, 'Pictures')

        initial_path = pictures if os.path.exists(pictures) else downloads if os.path.exists(downloads) else home

        content = FileChooserIconView(path=initial_path, filters=['*.jpg', '*.jpeg', '*.png', '*.bmp'])
        popup = Popup(title="Select an image file", content=content, size_hint=(0.9, 0.9), auto_dismiss=False)

        def on_submit(instance, selection, touch):
            if selection:
                popup.dismiss()
                self.load_and_predict(selection[0])

        content.bind(on_submit=on_submit)
        popup.open()

    def load_and_predict(self, filepath):
        try:
            pil_img = PILImage.open(filepath).convert('RGB')
            pil_img = ImageOps.exif_transpose(pil_img)

            image_np = np.array(pil_img)
            rgb_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            detections = self.mp_face_detection.process(rgb_img)

            if detections.detections:
                d = detections.detections[0]
                bboxC = d.location_data.relative_bounding_box
                ih, iw, _ = rgb_img.shape
                x1 = max(int(bboxC.xmin * iw) - 10, 0)
                y1 = max(int(bboxC.ymin * ih) - 10, 0)
                x2 = min(x1 + int(bboxC.width * iw) + 20, iw)
                y2 = min(y1 + int(bboxC.height * ih) + 20, ih)
                face_img = rgb_img[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = PILImage.fromarray(face_rgb).resize((IMG_SIZE, IMG_SIZE))
            else:
                face_pil = pil_img.resize((IMG_SIZE, IMG_SIZE))

            img_array = np.array(face_pil).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prob = self.model.predict(img_array)[0][0]
            label = "Female" if prob >= self.best_threshold else "Male"

            disp_img = np.array(pil_img)
            if detections.detections:
                cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            disp_img = PILImage.fromarray(disp_img).resize((280, 280))

            data = io.BytesIO()
            disp_img.save(data, format='png')
            data.seek(0)
            self.image_panel.texture = self.load_texture(data)

            self.result_label.text = f"Prediction: {label}"

        except UnidentifiedImageError:
            self.result_label.text = "Image format not supported."
        except Exception as e:
            self.result_label.text = f"Error: {e}"

    def load_texture(self, data):
        from kivy.core.image import Image as CoreImage
        return CoreImage(data, ext='png').texture

    def reset_app(self, instance):
        if self.live_mode:
            self.stop_live_cam()
        self.image_panel.texture = None
        self.result_label.text = "Upload an image or start live camera"

    def toggle_live_cam(self, instance):
        if not self.live_mode:
            self.start_live_cam()
        else:
            self.stop_live_cam()

    def start_live_cam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.text = "Cannot open camera."
            return
        self.live_mode = True
        self.live_btn.text = "Stop Live Cam"
        self.frame_count = 0
        Clock.schedule_interval(self.update_live_frame, 1.0 / 30.0)

    def stop_live_cam(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.live_mode = False
        self.live_btn.text = "Start Live Cam"
        self.image_panel.texture = None
        self.result_label.text = "Upload an image or start live camera"
        Clock.unschedule(self.update_live_frame)

    def update_live_frame(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            self.result_label.text = "Failed to grab frame."
            return
        
        frame = cv2.flip(frame, 1)

        self.frame_count += 1
        display_frame = frame.copy()

        if self.frame_count % PREDICTION_FREQUENCY == 0:
            small_frame = cv2.resize(frame, (320, 240))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_small)

            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240


            if results.detections:
                d = results.detections[0]
                bboxC = d.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = max(int(bboxC.xmin * 320 * scale_x) - 10, 0)
                y1 = max(int(bboxC.ymin * 240 * scale_y) - 10, 0)
                x2 = min(x1 + int(bboxC.width * 320 * scale_x) + 20, frame.shape[1])
                y2 = min(y1 + int(bboxC.height * 240 * scale_y) + 20, frame.shape[0])


                face = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                input_img = face_resized.astype('float32') / 255.0
                input_img = np.expand_dims(input_img, axis=0)

                prob = self.model.predict(input_img)[0][0]
                label = "Female" if prob >= self.best_threshold else "Male"

                self.last_label = label
                self.last_box = (x1, y1, x2, y2)
                self.last_prediction_time = time.time()
                self.result_label.text = f"Prediction: {label}"
            else:
                self.last_label = None
                self.last_box = None
                self.result_label.text = "No face detected."

        if self.last_label and self.last_box and (time.time() - self.last_prediction_time) < 1.5:
            x1, y1, x2, y2 = self.last_box
            text = f"{self.last_label}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        buf = cv2.flip(display_frame, 0).tobytes() 
        texture = self.image_panel.texture
        if not texture or texture.width != display_frame.shape[1] or texture.height != display_frame.shape[0]:
            from kivy.graphics.texture import Texture
            texture = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt='bgr')
            self.image_panel.texture = texture

        self.image_panel.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_panel.canvas.ask_update()

if __name__ == "__main__":
    GenderApp().run()
