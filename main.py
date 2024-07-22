import pygame
import pygame_gui
import tkinter as tk
from tkinter import filedialog
import subprocess
import sys
import os
import threading
from pygame_gui.core import ObjectID
from Process.twoStepDetection import detection

pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Video Processor Interface")

# Set up the manager for the GUI

manager = pygame_gui.UIManager((800, 600), "theme.json")

# Set up GUI elements
file_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((70, 50), (150, 50)), text='Choose File', manager=manager, object_id=ObjectID(class_id='@friendly_buttons',object_id='#hello_button'))
video_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((230, 50), (500, 50)), text="Video has not been selected", manager=manager)
process_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((70, 150), (150, 50)), text='Process', manager=manager, object_id=ObjectID(class_id='@friendly_buttons',object_id='#hello_button'))
message_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((230, 150), (500, 50)), text='Please select a video file', manager=manager)
open_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((70, 250), (150, 50)), text='Result', manager=manager, object_id=ObjectID(class_id='@friendly_buttons',object_id='#hello_button'))
result_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((230, 250), (500, 50)), text="No file result", manager=manager)

clock = pygame.time.Clock()
is_running = True
video_path = ""
flag = False
def open_video(file_path):
    try:
        if file_path:
            if sys.platform == "win32":
                subprocess.run(["start", file_path], check=True, shell=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path], check=True)
            else:
                subprocess.run(["xdg-open", file_path], check=True)
            message_label.set_text(f"Opening video file: {file_path}")
        else:
            message_label.set_text("No file selected.")
    except Exception as e:
        message_label.set_text(f"Failed to open video file: {str(e)}")

def open_file_dialog():
    global video_path
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    print(f"Selected video file: {video_path}")

def process_video_thread(video_path):
    global flag
    flag = False
    message_label.set_text("Processing...")
    detection(video_path)
    message_label.set_text("The video has been processed.")
    flag = True

while is_running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        if flag:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            result_label_text = f"{video_name}_result.mp4"
        else:
            result_label_text = "No file result"
        result_label.set_text(result_label_text)
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                # Loading video
                if event.ui_element == file_button:
                    flag = False
                    open_file_dialog()
                    if video_path:
                        video_label.set_text(video_path)
                        message_label.set_text("Successfully imported video")
                    else:
                        message_label.set_text("No file selected.")

                # Processing video
                if event.ui_element == process_button:
                    if video_path:
                        threading.Thread(target=process_video_thread, args=(video_path,)).start()
                    else:
                        message_label.set_text("Unable to process: no file selected!")
                # Play video processed
                if event.ui_element == open_button:
                    if video_path:
                        # Implement your video opening logic here
                        if flag:
                            video_name = os.path.splitext(os.path.basename(video_path))[0]
                            open_video(f"Results/{video_name}_result.mp4")
                    else:
                        print("No file selected.")

        manager.process_events(event)

    manager.update(time_delta)

    screen.fill((255, 255, 255))

    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
