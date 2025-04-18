import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStackedLayout, \
    QTextEdit, QDesktopWidget, QLineEdit, QDateEdit, QTimeEdit, QListWidget, QListWidgetItem, QSizePolicy, \
    QTableWidgetItem, QTableWidget, QHeaderView
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTime, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sqlite3


# Database utility
class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name

    def connect(self):
        return sqlite3.connect(self.db_name)

    def initialize(self):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                topic TEXT,
                module TEXT
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER,
                start_time TEXT,
                end_time TEXT,
                duration TEXT,
                FOREIGN KEY(class_id) REFERENCES classes(id)
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT,
                pose TEXT,
                attention_level FLOAT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )""")


def initialize_database():
    db = DatabaseManager('classes.db')
    db.initialize()
    print("Database initialized successfully")


# Video processing thread with error-handling
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, int)
    detected_class_signal = pyqtSignal(str, float, str)
    update_graph_signal = pyqtSignal(float)
    update_pose_graph_signal = pyqtSignal(str)

    def __init__(self, model_path, cam_id=0):
        super(VideoThread, self).__init__()
        self.cam_id = cam_id
        self.running = False
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def run(self):
        self.running = True
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.cam_id)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                try:
                    # Process the frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw landmarks if present
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        try:
                            pose = results.pose_landmarks.landmark
                            pose_row = list(np.array(
                                [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                 for landmark in pose]).flatten())
                            face = (results.face_landmarks.landmark
                                    if results.face_landmarks else [])
                            face_row = list(np.array(
                                [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                 for landmark in face]).flatten())
                            row = pose_row + face_row

                            X = pd.DataFrame([row])
                            body_language_class = self.model.predict(X)[0]
                            body_language_prob = self.model.predict_proba(X)[0]
                            self.display_class(image, results, body_language_class, body_language_prob)
                        except Exception as e:
                            print("Error processing pose landmarks:", e)

                    # Emit processed frame
                    self.change_pixmap_signal.emit(image, self.cam_id)

                except Exception as e:
                    print("Error in video thread processing:", e)

        cap.release()

    def display_class(self, image, results, body_language_class, body_language_prob):
        mp_holistic = mp.solutions.holistic
        coords = tuple(np.multiply(
            np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
            ),
            [640, 480]
        ).astype(int))

        # Updated attention detection logic based on new classes from the PKL file.
        if body_language_class.lower() == 'looking_straight':
            attention_status = "Focusing"
            attention_level = 1.0
        else:
            attention_status = "Not Focusing"
            attention_level = 0.0

        # Draw rectangle and display the detection on the image.
        cv2.rectangle(image,
                      (coords[0], coords[1] + 5),
                      (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                      (245, 117, 16), -1)
        cv2.putText(image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display attention status.
        cv2.putText(image, attention_status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display probability
        max_prob = max(body_language_prob)
        cv2.putText(image, f"Prob: {max_prob:.2f}", (coords[0], coords[1] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Emit signals for detected class, graph update, etc.
        self.detected_class_signal.emit(body_language_class, max_prob, attention_status)
        self.update_graph_signal.emit(attention_level)
        self.update_pose_graph_signal.emit(body_language_class)

    def stop(self):
        self.running = False


class HomePage(QWidget):
    def __init__(self, parent=None):
        super(HomePage, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        welcome_label = QLabel("Welcome to the Pose Detection App")
        welcome_label.setAlignment(Qt.AlignCenter)

        instructions_label = QLabel("Press 'Start' to begin detecting poses or 'Exit' to close the application.")
        instructions_label.setAlignment(Qt.AlignCenter)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.parent().show_scheduled_classes_page)
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.go_to_login_page)

        create_session_button = QPushButton("Create Session")
        create_session_button.clicked.connect(self.parent().show_create_session_page)

        historical_data_button = QPushButton("Historical Data")
        historical_data_button.clicked.connect(self.show_historical_data_page)

        layout.addWidget(welcome_label)
        layout.addWidget(instructions_label)
        layout.addWidget(start_button)
        layout.addWidget(create_session_button)
        layout.addWidget(historical_data_button)
        layout.addWidget(exit_button)

        self.setLayout(layout)

    def go_to_login_page(self):
        self.parent().back_to_login_page()

    def show_historical_data_page(self):
        self.parent().show_historical_data_page()


class CreateSessionPage(QWidget):
    def __init__(self, parent=None):
        super(CreateSessionPage, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()

        date_label = QLabel("Date:")
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)

        # Add time label + input
        time_label = QLabel("Time:")
        self.time_input = QTimeEdit()

        create_button = QPushButton("Create Session")
        create_button.clicked.connect(self.create_session)

        layout.addWidget(date_label)
        layout.addWidget(self.date_input)

        layout.addWidget(time_label)
        layout.addWidget(self.time_input)

        layout.addWidget(create_button)
        self.setLayout(layout)

    def create_session(self):
        date = self.date_input.date().toString(Qt.ISODate)
        # Grab time from QTimeEdit
        time = self.time_input.time().toString(Qt.ISODate)

        conn = sqlite3.connect('classes.db')
        c = conn.cursor()
        # Insert date/time, with empty placeholders for topic/module
        c.execute("INSERT INTO classes (date, time, topic, module) VALUES (?, ?, '', '')", (date, time))
        conn.commit()
        conn.close()

        # Show scheduled classes so user sees the new record
        self.parent().show_scheduled_classes_page()


class PoseDetectionPage(QWidget):
    def __init__(self, parent=None):
        super(PoseDetectionPage, self).__init__(parent)
        self.display_width = 1600
        self.display_height = 900
        self.setFixedSize(1600, 900)
        self.session_data = []
        self.attention_data = {0: 0, 1: 0}
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label2 = QLabel()

        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.image_label2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(200)

        self.back_button = QPushButton("Back to Home")
        self.back_button.clicked.connect(self.back_to_home)

        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.image_label2)
        left_layout.addWidget(self.console)
        left_layout.addWidget(self.back_button)

        right_layout = QVBoxLayout()
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.attention_percentage_label = QLabel("Attention: 0%")
        self.elapsed_time = QTime(0, 0, 0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)

        self.graph_canvas = GraphCanvas(self, width=5, height=4, dpi=100)
        self.pose_graph_canvas = PoseGraphCanvas(self, width=5, height=4, dpi=100)

        right_layout.addWidget(self.timer_label)
        right_layout.addWidget(self.graph_canvas)
        right_layout.addWidget(self.pose_graph_canvas)
        right_layout.addWidget(self.attention_percentage_label)

        self.thread1 = VideoThread('EmployeeDetection.pkl', cam_id=0)
        self.thread2 = VideoThread('EmployeeDetection.pkl', cam_id=1)
        self.thread1.change_pixmap_signal.connect(lambda img, cam=0: self.update_image(img, cam))
        self.thread2.change_pixmap_signal.connect(lambda img, cam=1: self.update_image(img, cam))
        self.thread1.detected_class_signal.connect(self.update_console)
        self.thread2.detected_class_signal.connect(self.update_console)
        self.thread1.update_graph_signal.connect(lambda level: self.graph_canvas.update_graph(level, 0))
        self.thread2.update_graph_signal.connect(lambda level: self.graph_canvas.update_graph(level, 1))
        self.thread1.update_graph_signal.connect(lambda level: self.update_attention_data(level, 0))
        self.thread2.update_graph_signal.connect(lambda level: self.update_attention_data(level, 1))
        self.thread1.update_pose_graph_signal.connect(self.pose_graph_canvas.update_graph)
        self.thread2.update_pose_graph_signal.connect(self.pose_graph_canvas.update_graph)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.thread = None

    def receive_attention_data(self, attention_data):
        for cam_id, level in attention_data.items():
            self.update_attention_data(level, cam_id)

    def update_attention_data(self, attention_level, cam_id):
        self.attention_data[cam_id] = attention_level
        self.calculate_and_display_attention_percentage()

    def calculate_and_display_attention_percentage(self):
        total_cameras = len(self.attention_data)
        if total_cameras == 0:
            return
        paying_attention = sum(self.attention_data.values())
        percent_paying_attention = (paying_attention / total_cameras) * 100
        self.attention_percentage_label.setText(f"Attention: {percent_paying_attention:.0f}%")

    def start_video_streams(self):
        self.thread1.start()
        self.thread2.start()

    def update_timer(self):
        self.elapsed_time = self.elapsed_time.addSecs(1)
        self.timer_label.setText(self.elapsed_time.toString("hh:mm:ss"))

    def start_timer(self):
        self.elapsed_time = QTime(0, 0, 0)
        self.timer.start(1000)

    def stop_timer(self):
        self.timer.stop()
        self.timer_label.setText("00:00:00")

    def collect_data(self, detected_class, probability, attention_status):
        current_time = QTime.currentTime().toString("hh:mm:ss")
        self.session_data.append({
            'timestamp': current_time,
            'pose': detected_class,
            'attention_level': probability
        })

    def update_console(self, detected_class, probability, attention_status):
        self.collect_data(detected_class, probability, attention_status)
        self.console.append(
            f"Detected: {detected_class}, Probability: {probability:.2f}, Attention: {attention_status}")

    def stop_video_streams(self):
        if hasattr(self, 'thread1') and self.thread1.isRunning():
            self.thread1.stop()
            self.thread1.wait()
        if hasattr(self, 'thread2') and self.thread2.isRunning():
            self.thread2.stop()
            self.thread2.wait()

    def stop_session(self):
        self.stop_video_streams()
        self.stop_timer()
        if hasattr(self, 'class_id') and self.class_id:
            self.save_session_data_to_database(self.class_id)
            self.parent().scheduled_classes_page.remove_class(self.class_id)
            self.parent().historical_data_page.load_sessions()

    def save_session_data_to_database(self, class_id):
        end_time = QTime.currentTime().toString("hh:mm:ss")
        duration = self.elapsed_time.toString("hh:mm:ss")
        conn = sqlite3.connect('classes.db')
        c = conn.cursor()
        c.execute("INSERT INTO sessions (class_id, start_time, end_time, duration) VALUES (?, ?, ?, ?)",
                  (class_id, self.start_time, end_time, duration))
        session_id = c.lastrowid

        for data_point in self.session_data:
            c.execute("INSERT INTO session_data (session_id, timestamp, pose, attention_level) VALUES (?, ?, ?, ?)",
                      (session_id, data_point['timestamp'], data_point['pose'], data_point['attention_level']))

        conn.commit()
        conn.close()
        print(f"Session {session_id} for class {class_id} saved successfully.")

    def back_to_home(self):
        self.stop_session()
        self.parent().show_home_page()

    def back_to_dashboard(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
        self.stop_timer()
        self.parent().back_to_dashboard()

    def update_image(self, cv_img, cam_id):
        qt_img = self.convert_cv_qt(cv_img)
        if cam_id == 0:
            self.image_label.setPixmap(qt_img)
        elif cam_id == 1:
            self.image_label2.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width // 2, self.display_height // 2, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor('#f0f0f0')
        super(GraphCanvas, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.plot_data1 = []
        self.plot_data2 = []

    def update_graph(self, attention_level, cam_id):
        if cam_id == 0:
            self.plot_data1.append(attention_level)
        elif cam_id == 1:
            self.plot_data2.append(attention_level)

        self.axes.clear()
        x1 = np.arange(len(self.plot_data1))
        x2 = np.arange(len(self.plot_data2))

        # A little polynomial smoothing
        if len(self.plot_data1) > 1:
            coefs1 = np.polyfit(x1, self.plot_data1, deg=min(5, len(x1) - 1))
            p1 = np.poly1d(coefs1)
            self.axes.plot(x1, p1(x1), 'r-', label='Camera 1')

        if len(self.plot_data2) > 1:
            coefs2 = np.polyfit(x2, self.plot_data2, deg=min(5, len(x2) - 1))
            p2 = np.poly1d(coefs2)
            self.axes.plot(x2, p2(x2), 'b-', label='Camera 2')

        self.axes.set_title("Attention Level Over Time")
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("Attention Level")
        self.axes.legend()
        self.axes.grid(True)
        self.draw()

    def reset_graph(self):
        self.plot_data1 = []
        self.plot_data2 = []


class PoseGraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super(PoseGraphCanvas, self).__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.poses_data = {}

    def update_graph(self, pose_class):
        if pose_class in self.poses_data:
            self.poses_data[pose_class] += 1
        else:
            self.poses_data[pose_class] = 1

        self.axes.clear()
        poses, counts = zip(*sorted(self.poses_data.items())) if self.poses_data else ([], [])
        self.axes.bar(poses, counts, color='g')
        self.axes.set_title('Detected Poses Count')
        self.axes.set_xlabel('Pose')
        self.axes.set_ylabel('Count')
        self.axes.set_xticks(range(len(poses)))
        self.axes.set_xticklabels(poses, rotation=45, ha="right")
        self.figure.tight_layout()
        self.draw()


class LoginPage(QWidget):
    def __init__(self, parent=None):
        super(LoginPage, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        welcome_label = QLabel("Login to Pose Detection App")
        welcome_label.setAlignment(Qt.AlignCenter)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)

        login_button = QPushButton("Login")
        login_button.clicked.connect(self.login)

        self.error_label = QLabel()
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: red")

        layout.addWidget(welcome_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addWidget(login_button)
        layout.addWidget(self.error_label)
        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username == "admin" and password == "admin":
            self.parent().show_home_page()
        else:
            self.error_label.setText("Incorrect username or password")


class ScheduledClassesPage(QWidget):
    def __init__(self, parent=None):
        super(ScheduledClassesPage, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.class_clicked)
        self.load_classes()

        button_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.back_to_home_page)
        delete_all_button = QPushButton("Delete All Sessions")
        delete_all_button.clicked.connect(self.delete_all_sessions)

        button_layout.addWidget(back_button)
        button_layout.addWidget(delete_all_button)

        layout.addWidget(self.class_list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_classes(self):
        self.class_list.clear()
        conn = sqlite3.connect('classes.db')
        c = conn.cursor()
        # Now only selecting id, date, time – removing topic and module
        c.execute("SELECT id, date, time FROM classes")
        classes = c.fetchall()
        for class_data in classes:
            item = QListWidgetItem(f"ID: {class_data[0]}, Date: {class_data[1]}, Time: {class_data[2]}")
            self.class_list.addItem(item)
        conn.close()

    def class_clicked(self, item):
        class_id = int(item.text().split(',')[0].split(': ')[1])
        self.parent().start_pose_detection(class_id)

    def remove_class(self, class_id):
        for index in range(self.class_list.count()):
            item = self.class_list.item(index)
            if class_id == int(item.text().split(',')[0].split(': ')[1]):
                self.class_list.takeItem(index)
                break

    def delete_all_sessions(self):
        conn = sqlite3.connect('classes.db')
        c = conn.cursor()
        c.execute("DELETE FROM session_data")
        c.execute("DELETE FROM sessions")
        c.execute("DELETE FROM classes")
        conn.commit()
        conn.close()
        self.load_classes()

    def back_to_home_page(self):
        self.parent().show_home_page()


class HistoricalDataPage(QWidget):
    def __init__(self, parent=None):
        super(HistoricalDataPage, self).__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self.session_clicked)

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.goBack)

        layout.addWidget(self.session_list)
        layout.addWidget(back_button)
        self.setLayout(layout)
        self.load_sessions()

    def load_sessions(self):
        self.session_list.clear()
        conn = sqlite3.connect('classes.db')
        c = conn.cursor()
        c.execute("SELECT session_id, class_id, start_time, end_time, duration FROM sessions")
        sessions = c.fetchall()
        for session in sessions:
            # We'll fetch date/time from classes if we like, but we no longer show topic/module
            c.execute("SELECT date, time FROM classes WHERE id = ?", (session[1],))
            row = c.fetchone()
            if row:
                date_str = row[0]
                time_str = row[1]
            else:
                date_str = "N/A"
                time_str = "N/A"
            item = QListWidgetItem(f"Session ID: {session[0]}, Date: {date_str}, Time: {time_str}, "
                                   f"Start: {session[2]}, End: {session[3]}, Duration: {session[4]}")
            self.session_list.addItem(item)
        conn.close()

    def session_clicked(self, item):
        session_id = int(item.text().split(',')[0].split(': ')[1])
        self.parent().show_session_details(session_id)

    def goBack(self):
        self.parent().show_home_page()


class SessionDetailsPage(QWidget):
    def __init__(self, session_id, parent=None):
        super(SessionDetailsPage, self).__init__(parent)
        self.session_id = session_id
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        conn = sqlite3.connect('classes.db')
        c = conn.cursor()

        # Fetch session info
        c.execute("SELECT class_id, start_time, end_time, duration FROM sessions WHERE session_id = ?",
                  (self.session_id,))
        session_info = c.fetchone()
        class_id = session_info[0]
        start_time = session_info[1]
        end_time = session_info[2]
        duration = session_info[3]

        # Instead of topic/module, show the date/time from the classes table
        c.execute("SELECT date, time FROM classes WHERE id = ?", (class_id,))
        row = c.fetchone()
        class_date = row[0] if row else "N/A"
        class_time = row[1] if row else "N/A"

        layout.addWidget(QLabel(f"Class Date: {class_date}"))
        layout.addWidget(QLabel(f"Class Time: {class_time}"))
        layout.addWidget(QLabel(f"Session Start: {start_time}"))
        layout.addWidget(QLabel(f"Session End: {end_time}"))
        layout.addWidget(QLabel(f"Session Duration: {duration}"))

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(["Timestamp", "Pose", "Attention Level", "Additional Info"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Now fetch the actual data from session_data
        c.execute("SELECT timestamp, pose, attention_level FROM session_data WHERE session_id = ?", (self.session_id,))
        data_points = c.fetchall()
        for data in data_points:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(data[0]))
            self.tableWidget.setItem(row_position, 1, QTableWidgetItem(data[1]))
            self.tableWidget.setItem(row_position, 2, QTableWidgetItem(f"{data[2]:.2f}"))

        conn.close()

        layout.addWidget(self.tableWidget)

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.goBack)
        layout.addWidget(back_button)

        self.setLayout(layout)

    def goBack(self):
        self.parent().show_historical_data_page()


class App(QWidget):
    def __init__(self):
        super(App, self).__init__()
        self.setWindowTitle("Pose Detection with PyQt")
        self.display_width = 1280
        self.display_height = 720
        initialize_database()
        self.initUI()

    def initUI(self):
        self.stacked_layout = QStackedLayout(self)

        self.login_page = LoginPage(self)
        self.home_page = HomePage(self)
        self.create_session_page = CreateSessionPage(self)
        self.scheduled_classes_page = ScheduledClassesPage(self)
        self.pose_detection_page = PoseDetectionPage(self)
        self.historical_data_page = HistoricalDataPage(self)
        self.session_details_page = None

        self.stacked_layout.addWidget(self.login_page)
        self.stacked_layout.addWidget(self.home_page)
        self.stacked_layout.addWidget(self.create_session_page)
        self.stacked_layout.addWidget(self.scheduled_classes_page)
        self.stacked_layout.addWidget(self.pose_detection_page)
        self.stacked_layout.addWidget(self.historical_data_page)

        self.setLayout(self.stacked_layout)
        self.center()

    def show_login_page(self):
        self.stacked_layout.setCurrentWidget(self.login_page)

    def back_to_login_page(self):
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()
        self.stacked_layout.setCurrentWidget(self.login_page)

    def show_home_page(self):
        self.stacked_layout.setCurrentWidget(self.home_page)

    def show_create_session_page(self):
        self.stacked_layout.setCurrentWidget(self.create_session_page)

    def show_scheduled_classes_page(self):
        self.scheduled_classes_page.load_classes()
        self.stacked_layout.setCurrentWidget(self.scheduled_classes_page)

    def start_pose_detection(self, class_id):
        self.pose_detection_page.class_id = class_id
        self.pose_detection_page.start_time = QTime.currentTime().toString("hh:mm:ss")

        # Start threads if they aren't already running
        if not self.pose_detection_page.thread1.isRunning():
            self.pose_detection_page.thread1.start()
        if not self.pose_detection_page.thread2.isRunning():
            self.pose_detection_page.thread2.start()

        # Start timer
        self.pose_detection_page.start_timer()
        self.stacked_layout.setCurrentWidget(self.pose_detection_page)

    def back_to_dashboard(self):
        if self.pose_detection_page.thread:
            self.pose_detection_page.thread.stop()
            self.pose_detection_page.thread.wait()
            self.pose_detection_page.graph_canvas.reset_graph()
            self.pose_detection_page.stop_session(self.pose_detection_page.class_id)
            selected_class_item = self.scheduled_classes_page.class_list.selectedItems()
            if selected_class_item:
                self.scheduled_classes_page.class_list.takeItem(
                    self.scheduled_classes_page.class_list.row(selected_class_item[0])
                )
        self.stacked_layout.setCurrentWidget(self.home_page)

    def show_historical_data_page(self):
        self.historical_data_page.load_sessions()
        self.stacked_layout.setCurrentWidget(self.historical_data_page)

    def show_session_details(self, session_id):
        self.session_details_page = SessionDetailsPage(session_id, self)
        self.stacked_layout.addWidget(self.session_details_page)
        self.stacked_layout.setCurrentWidget(self.session_details_page)

    def center(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.display_width) // 2
        y = (screen_geometry.height() - self.display_height) // 2
        self.move(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    sys.exit(app.exec_())
