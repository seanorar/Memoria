from Tkinter import *
import Tkinter as tk
from PIL import ImageTk, Image
from tkFileDialog import askopenfilename
from code_m import get_img_roi

proto_val = "prototxt"
model_val = "caffemodel"
img_val = "image"

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))

def load_file(msg_label):
    Tk().withdraw()
    filename = askopenfilename()
    str_split = filename.split("/")
    msg_label.config(text=str_split[len(str_split)-1])
    return filename

def set_proto(label):
    global proto_val
    proto_val = load_file(label)

def set_model(label):
    global model_val
    model_val = load_file(label)

def set_img(label):
    global img_val
    img_val = load_file(label)

def create_img_roi():
    global canvas
    global proto_val
    global model_val
    global img_val
    get_img_roi(img_val,proto_val,model_val)
    img = ImageTk.PhotoImage(Image.open("resultado.jpg"))
    window = tk.Toplevel()
    canvas = Canvas(window, width=900, height=500)
    canvas.pack(side=tk.LEFT)

    scrollbar = tk.Scrollbar(window, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill='y')

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.bind('<Configure>', on_configure)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='nw')
    canvas.img = img

    l = Label(frame, image=img)
    l.pack(side="left")



root = Tk()
proto_frame = Frame(root)
proto_frame.pack()
proto_label = Message(proto_frame, text = proto_val, width=100)
proto_buttom = Button(proto_frame,text="select net model",
                      command= lambda : set_proto(proto_label)).pack(side="left")
proto_label.pack()

model_frame = Frame(root)
model_frame.pack()
model_label = Message(model_frame, text = model_val, width=100)
model_buttom = Button(model_frame,
                         text="selecet caffe model", fg="black",
                         command= lambda : set_model(model_label)).pack(side="left")
model_label.pack()


img_frame = Frame(root)
img_frame.pack()
img_label = Message(img_frame, text = img_val, width=100)
img_buttom = Button(img_frame,
                         text="select img", fg="black",
                         command= lambda : set_img(img_label)).pack(side="left")
img_label.pack()

action_frame = Frame(root)
action_frame.pack()
action_buttom = Button(action_frame,
                         text="process", fg="black",
                         command= create_img_roi ).pack(side="left")
img_label.pack()


root.mainloop()