# http://stackoverflow.com/questions/17296140/handwritten-english-character-data-set-where-to-get-and-openly-available
import Tkinter as tk
import random
import os

from PIL import Image
from PIL import ImageTk
import numpy as np

import dataLoader
import SimpleNeuralNetworks


class DIP(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.training_data, self.validation_data, self.test_data = dataLoader.load_data_wrapper()
        self.imageToNumberNet = SimpleNeuralNetworks.StaticNetwork([784, 30, 10])
        self.numberToImageNet = SimpleNeuralNetworks.StaticNetwork([10, 150, 784])
        try:
            fileAppendage = '-'.join([str(x) for x in self.imageToNumberNet.nodes])
            fp = (os.path.dirname(os.path.realpath(__file__)) + "/networkData"+fileAppendage+".pkl")
            self.imageToNumberNet.loadNetwork(fp)  # Try loading network if it exists
        except:
            pass
        self.correctness = []
        self.init_UI()
        self.update_image()

    def init_UI(self):
        self.parent.title("NN Image Scanner")
        self.pack(fill=tk.BOTH, expand=1)

        self.fileImageLabel = tk.Label(self, border=25)
        self.fileImageLabel.grid(row=1, column=1)

        self.guessLabelText = tk.StringVar()
        self.guessLabel = tk.Label(self, textvariable=self.guessLabelText, font=("Helvetica", 24))
        self.guessLabel.grid(row=1, column=2)

        self.actualLabelText = tk.StringVar()
        self.actualLabel = tk.Label(self, textvariable=self.actualLabelText, font=("Helvetica", 18))
        self.actualLabel.grid(row=2, column=1)

        label = tk.Label(self, text="Reverse Image Generation", font=("Helvetica", 18))
        label.grid(row=3, column=1)

        self.generatedImageLabel = tk.Label(self, border=25)
        self.generatedImageLabel.grid(row=3, column=2)

    def scan_image_from_data(self, data):
        return np.argmax(self.imageToNumberNet.feedForward(data[0]))

    def image_from_data(self, data):
        img = Image.new('L', (28, 28), "black") # create a new black image
        pixels = img.load() # create the pixel map
        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                pixels[i,j] = (1 - data[j * 28 + i]) * 255 # set the colour accordingly
        img = img.resize((28 * 5, 28 * 5), Image.ANTIALIAS)
        return img


    def update_image(self):
        #Train Nets
        trainAmount = 10
        trainStart = random.randint(0, len(self.test_data) - trainAmount)
        self.imageToNumberNet.train(self.training_data[trainStart:trainStart+trainAmount], trainAmount, 3.0, None, False)
        rtr_d = []
        for i in range(trainStart, trainStart + trainAmount):
            rtr_d.append([self.imageToNumberNet.feedForward(self.training_data[i][0]), self.training_data[i][0]])
        self.numberToImageNet.train(rtr_d, trainAmount, 3.0, None, False)
        data = self.test_data[random.randint(0, len(self.test_data) - 1)] #Pick a random image
        img = ImageTk.PhotoImage(self.image_from_data(data[0]))
        self.fileImageLabel.configure(image = img)
        self.fileImageLabel.image = img

        real = data[1]
        guess = self.scan_image_from_data(data)
        if (real == guess):
            self.correctness.append(1)
        else:
            self.correctness.append(0)
        if (len(self.correctness) > 100):
            self.correctness.pop(0)
        print "\r" * 100 + "Image to number Accuracy " + str(sum(self.correctness) / float(len(self.correctness))),
        self.guessLabelText.set("Guess: "+str(guess))
        self.actualLabelText.set(str(real))

        ## Number to image
        d = np.zeros((10, 1))
        d[data[1]] = 1.0 #Only stimulates the neuron of the number we have
        reverseImageData = self.numberToImageNet.feedForward(d) #Feed the number to the image generator
        for i in range(len(reverseImageData)):
            reverseImageData[i] = reverseImageData[i][0] #Remove each value from its singular array
        img2 = ImageTk.PhotoImage(self.image_from_data(reverseImageData)) #image_from_data is expecting an array of image data and
        self.generatedImageLabel.configure(image = img2)
        self.generatedImageLabel.image = img2

        #save
        if (random.randint(0, 100) == 1):
            fileAppendage = '-'.join([str(x) for x in self.imageToNumberNet.nodes])
            fp = (os.path.dirname(os.path.realpath(__file__)) + "/networkData"+fileAppendage+".pkl")
            self.imageToNumberNet.saveNetwork(fp)
            self.after(500, self.update_image)
        else:
            self.after(1, self.update_image)

def main():

    root = tk.Tk()
    DIP(root)
    root.geometry("500x450")
    root.mainloop()


if __name__ == '__main__':

    '''net1 = SimpleNeuralNetworks.SimpleNeuralNetworks([784, 30, 10])
    net = SimpleNeuralNetworks.SimpleNeuralNetworks([10, 300, 784])
    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    rtr_d = [[0 for z in xrange(2)] for y in xrange(len(test_data))]
    time.sleep(10)
    for i in range(len(test_data)):
        rtr_d[i][0], rtr_d[i][1] = net1.feedForward(training_data[i][0]), training_data[i][0]

    net.train(rtr_d, 10, 3.0, None)'''

    main()
