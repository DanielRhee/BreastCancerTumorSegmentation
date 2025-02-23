from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from Models import predict

root = Tk()
root.title("Breast Cancer Tumor Segmentation")
model = predict.predictor()

def save(result_image):
    file = filedialog.asksaveasfile(defaultextension=".png", 
                                     filetypes=[("png", "*.png")
                                     ])
    result_image.save(file)

def open():
    #entry_frame is a row with the png and associated buttons
    entry_frame = Frame(root)
    image_path = filedialog.askopenfilename(initialdir="~", title="upload ultra sound image", filetypes=(("png", "*.png"), ("all files", "*,*")))
    original_image = ImageTk.PhotoImage(Image.open(image_path))
    result_image, classification = model.predict(Image.open(image_path).convert("RGB"))
    #this turns the image into a widget so it can be displayed
    result_label = Label(entry_frame)
    save_btn = Button(entry_frame, test= "save", command=save(result_image))
    #TODO add a delete button. this will likely include tracking and deleting the frame
    #the image and save button are packed onto a frame and then the frame is packed into the window
    result_label.pack()
    save_btn.pack()
    entry_frame.pack()

upload_btn = Button(root, text= "upload file", command= open()).pack()

def main():
    root.mainloop()

if __name__ == "__main__":
    root.mainloop()


