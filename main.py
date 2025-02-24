from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from Models import predict

model = predict.predictor()

def save(result_image):
    file = filedialog.asksaveasfile(defaultextension=".png", 
                                     filetypes=[("png", "*.png")
                                     ])
    result_image.save(file)

def openImage():
    #entry_frame is a row with the png and associated buttons
    image_path = filedialog.askopenfilename(initialdir="~", title="upload ultra sound image", filetypes=(("png", "*.png"), ("all files", "*,*")))
    pillow_image = Image.open(image_path)
    result_image, classification_num = model.predict(pillow_image.convert("RGB"))

    #this turns the image into a widget so it can be displayed
    classification_key = {0:"beign", 1: "malignant", 2:"normal"}
    classification = classification_key[classification_num]
    result_tkimage = ImageTk.PhotoImage(result_image)
    result_label = Label(scrollable_frame, image=result_tkimage, text=classification, compound=TOP)
    result_label.image = result_tkimage
    #save_btn = Button(entry_frame, test= "save", command=save)
    #TODO add a delete button. this will likely include tracking and deleting the frame
    #the image and save button are packed onto a frame and then the frame is packed into the window
    result_label.pack()
    #save_btn.pack()
    #entry_frame.pack()



def main():
    root.mainloop()

if __name__ == "__main__":
    root = Tk()
    root.title("Breast Cancer Tumor Segmentation")
    upload_btn = Button(root, text= "upload file", command= openImage).pack()
    save_btn = Button(root, text= "Save Results")
    canvas = Canvas(root)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar_y = Scrollbar(root, orient=VERTICAL, command=canvas.yview)
    scrollbar_y.pack(side=RIGHT, fill=Y)

    scrollbar_x = Scrollbar(canvas, orient=HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=BOTTOM, fill=X)

    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    
    scrollable_frame = Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main()
    
   


