import tkinter as tk
from tkinter import filedialog
import customtkinter
import os
from PIL import Image, ImageTk, ImageOps
import cv2
import glob
from ovvas import do_ovaas


import threading

FONT_TYPE = "meiryo"

class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        # メンバー変数の設定
        self.fonts = (FONT_TYPE, 15)
        self.csv_filepath = None
        self.movie_path = None
        
        #動画再生コントロール用
        self.set_movie = True
        self.thread_set = False
        self.start_movie = False
        self.video_frame = None
        self.generated_video_path = None
        
        # Entry()を使う場合文字列を操作することが多く、一般的にはStringVarを使う。
        # StringVarを運用させれば、後からエントリの文字列が変わっても、StringVarの値も変わる（同期されている）
        self.selected_option = tk.StringVar() 

        # 保存ディレクトリの設定
        self.output_dir = os.path.abspath(os.path.dirname(__file__))

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # CustomTkinter のフォームデザイン設定
        customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        self.title("OVaaS")

        # フォームサイズ設定
        self.geometry("700x480")
        self.minsize(350, 400)

        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(1, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)


        # ファイルパスを指定するテキストボックス。これだけ拡大したときに、幅が広がるように設定する。
        self.textbox = customtkinter.CTkEntry(master=self, placeholder_text="ファイルを読み込む", width=120, font=self.fonts)
        self.textbox.grid(row=0, column=0, padx=10, pady=(0,10), sticky="nwe")
        # row=行, column=列, padx=外側の横の隙間を指定, pady=外側の縦の隙間を指定
        # sticky=座標の基準位置を指定  上＝N, 下＝S, 右＝E, 左＝W

        # ファイル選択ボタン
        self.button_select = customtkinter.CTkButton(master=self, 
            fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),   # ボタンを白抜きにする
            command=self.button_select_callback, text="ファイル選択", font=self.fonts)
        self.button_select.grid(row=0, column=1, padx=10, pady=(0,10))
        
        # 実行ボタン
        self.button_open = customtkinter.CTkButton(master=self, command=self.button_open_callback, text="実行", font=self.fonts)
        self.button_open.grid(row=0, column=2, padx=10, pady=(0,10))
        
        # 停止ボタン
        self.button_open = customtkinter.CTkButton(master=self, command=self.button_stop_callback, text="停止", font=self.fonts)
        self.button_open.grid(row=0, column=3, padx=10, pady=(0,10))
        

        # ラジオボックス
        self.radiobutton_age = customtkinter.CTkRadioButton(master=self, text="action_recognition", variable=self.selected_option, value="face-age-estimation", command=self.on_radiobox_select)
        self.radiobutton_age.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ne")

        self.radiobutton_pose = customtkinter.CTkRadioButton(master=self, text="pose-estimation", variable=self.selected_option, value="pose-estimation", command=self.on_radiobox_select)
        self.radiobutton_pose.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="n")

        self.radiobutton_object = customtkinter.CTkRadioButton(master=self, text="object-detection", variable=self.selected_option, value="object-detection", command=self.on_radiobox_select)
        self.radiobutton_object.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="nw")

        # 生成された動画を表示するキャンバスの作成
        self.canvas = tk.Canvas(master=self, width=640, height=480)
        self.canvas.grid(row=2, columnspan=4, padx=10, pady=(0, 10))

    '''
        # 保存ボタンを置く
        self.button_save = customtkinter.CTkButton(master=self, command=self.button_save_callback, text="保存", font=self.fonts)
        self.button_save.grid(row=2, column=2, padx=10, pady=(0,10), sticky="s")   
    '''

        

    def button_select_callback(self):
        """
        選択ボタンが押されたときのコールバック。ファイル選択ダイアログを表示する
        """
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(1, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        current_dir = os.path.abspath(os.path.dirname(__file__))
        self.movie_path = tk.filedialog.askopenfilename(filetypes=[("MP4ファイル", ("*.MP4", "*.BIN", "*.XML"))],initialdir=current_dir)

        # ファイルパスをテキストボックスに記入
        self.textbox.delete(0, tk.END)
        self.textbox.insert(0, self.movie_path)
        
        # Movie standby.
        self.set_movie = True
        self.thread_set = True
        self.thread_main = threading.Thread(target=self.main_thread_func)
        self.thread_main.start()

    def on_radiobox_select(self):
        """
        ラジオボックスの選択が変更された際のコールバック
        """
        self.result = self.selected_option.get()  # 選択されたオプションを取得
        
    def button_open_callback(self):
        """
        実行ボタンが押された時のコールバック
        """
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.start_movie = True

        
        file_path = self.textbox.get()
        model_name=self.result
        do_ovaas(file_path, model_name)
        
        
    def button_stop_callback(self):
        self.start_movie = False
        self.set_movie = False
    
    def main_thread_func(self):

        self.list_of_files = glob.glob('/Users/oosakiharuna/Documents/ovaas-main/result')

        if not self.list_of_files:
            print("No files found in the specified directory.")
            return 
        self.lastes_file = max(self.list_of_files, key=os.path.getctime)
        
        self.video_cap = cv2.VideoCapture(self.movie_path)
        ret, self.video_frame, *_ = self.lastes_file


        if self.video_frame is None:
            print("None")

        while self.set_movie:

            if self.start_movie:


                ret, self.video_frame = self.video_cap.read()

                if ret:
                    # convert color order from BGR to RGB
                    pil = self.cvtopli_color_convert(self.video_frame)

                    self.effect_img, self.canvas_create = self.resize_image(
                        pil, self.canvas
                    )
                    self.replace_canvas_image(
                        self.effect_img, self.canvas, self.canvas_create
                    )
                else:
                    self.start_movie = False

    
    
    def replace_canvas_image(self, pic_img, canvas_name, canvas_name_create):
        canvas_name.photo = ImageTk.PhotoImage(pic_img)
        canvas_name.itemconfig(canvas_name_create, image=canvas_name.photo)

    def cvtopli_color_convert(self, video):
        rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def resize_image(self, img, canvas):

        w = img.width
        h = img.height
        w_offset = 250 - (w * (500 / h) / 2)
        h_offset = 250 - (h * (700 / w) / 2)

        if w > h:
            resized_img = img.resize((int(w * (700 / w)), int(h * (700 / w))))
        else:
            resized_img = img.resize((int(w * (500 / h)), int(h * (500 / h))))

        self.pil_img = ImageTk.PhotoImage(resized_img)
        canvas.delete("can_pic")

        if w > h:
            resized_img_canvas = canvas.create_image(
                0, h_offset, anchor="nw", image=self.pil_img, tag="can_pic"
            )

        else:
            resized_img_canvas = canvas.create_image(
                w_offset, 0, anchor="nw", image=self.pil_img, tag="can_pic"
            )

        return resized_img, resized_img_canvas

            
        


if __name__ == "__main__":
    app = App()
    app.mainloop()
