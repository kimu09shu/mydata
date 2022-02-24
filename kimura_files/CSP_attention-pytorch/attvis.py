import cv2
import tkinter
from PIL import Image, ImageTk

root = tkinter.Tk()
root.title("Attention map")

image_bgr = cv2.imread("frankfurt_000000_000294_leftImg8bit.png")
image_bgr = cv2.resize(image_bgr, (int(image_bgr.shape[1]/4), int(image_bgr.shape[0]/4)))
image_bgr = cv2.copyMakeBorder(image_bgr, 7, 7, 7, 7, cv2.BORDER_CONSTANT, (0, 0, 0))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
image_tk = ImageTk.PhotoImage(image_pil)

canvas = tkinter.Canvas(root, width=2000, height=1000)
canvas.pack()
canvas.create_image(0, 0, image=image_tk, anchor='nw')

root.mainloop()
