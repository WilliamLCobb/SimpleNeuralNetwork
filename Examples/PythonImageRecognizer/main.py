# http://stackoverflow.com/questions/17296140/handwritten-english-character-data-set-where-to-get-and-openly-available
import Tkinter as tk
import random

from PIL import Image
from PIL import ImageTk
import numpy as np

import dataLoader
import SimpleNeuralNetwork



class DIP(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.training_data, self.validation_data, self.test_data = dataLoader.load_data_wrapper()
        self.imageToNumberNet = SimpleNeuralNetwork.NeuralNetwork([784, 30, 10])
        self.numberToImageNet = SimpleNeuralNetwork.NeuralNetwork([10, 150, 784])
        try:
            self.imageToNumberNet.loadNetwork()
        except:
            pass
        self.correctness = []
        self.initUI()
        self.updateImage()

    def initUI(self):
        self.parent.title("NN Image Scanner")
        self.pack(fill = tk.BOTH, expand = 1)

        self.fileImageLabel = tk.Label(self, border = 25)
        self.fileImageLabel.grid(row = 1, column = 1)

        self.guessLabelText = tk.StringVar()
        self.guessLabel = tk.Label(self, textvariable=self.guessLabelText, font=("Helvetica", 24))
        self.guessLabel.grid(row = 1, column = 2)

        self.actualLabelText = tk.StringVar()
        self.actualLabel = tk.Label(self, textvariable=self.actualLabelText, font=("Helvetica", 18))
        self.actualLabel.grid(row = 2, column = 1)

        label = tk.Label(self, text="Reverse Image Generation", font=("Helvetica", 18))
        label.grid(row = 3, column = 1)

        self.generatedImageLabel = tk.Label(self, border = 25)
        self.generatedImageLabel.grid(row = 3, column = 2)


    def scanImageFromData(self, data):
        return np.argmax(self.imageToNumberNet.feedForward(data[0]))


    def imageFromData(self, data):
        img = Image.new( 'L', (28, 28), "black") # create a new black image
        pixels = img.load() # create the pixel map
        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                pixels[i,j] = (1 - data[j * 28 + i]) * 255 # set the colour accordingly
        img = img.resize((28 * 5, 28 * 5), Image.ANTIALIAS)
        return img


    def updateImage(self):
        #Train Nets
        trainAmount = 10
        trainStart = random.randint(0, len(self.test_data) - trainAmount)
        self.imageToNumberNet.train(self.training_data[trainStart:trainStart+trainAmount], trainAmount, 3.0, None, False)
        rtr_d = []
        for i in range(trainStart, trainStart + trainAmount):
            rtr_d.append([self.imageToNumberNet.feedForward(self.training_data[i][0]), self.training_data[i][0]])
        self.numberToImageNet.train(rtr_d, trainAmount, 3.0, None, False)
        data = self.test_data[random.randint(0, len(self.test_data) - 1)] #Pick a random image
        img = ImageTk.PhotoImage(self.imageFromData(data[0]))
        self.fileImageLabel.configure(image = img)
        self.fileImageLabel.image = img

        real = data[1]
        guess = self.scanImageFromData(data)
        if (real == guess):
            self.correctness.append(1)
        else:
            self.correctness.append(0)
        if (len(self.correctness) > 100):
            self.correctness.pop(0)
        print "\r" * 100 + "Accuracy " + str(sum(self.correctness) / float(len(self.correctness))),
        self.guessLabelText.set("Guess: "+str(guess))
        self.actualLabelText.set(str(real))

        ## Number to image
        d = np.zeros((10, 1))
        d[data[1]] = 1.0 #Only stimulates the neuron of the number we have
        reverseImageData = self.numberToImageNet.feedForward(d) #Feed the number to the image generator
        for i in range(len(reverseImageData)):
            reverseImageData[i] = reverseImageData[i][0] #Remove each value from its singular array
        img2 = ImageTk.PhotoImage(self.imageFromData(reverseImageData)) #imageFromData is expecting an array of image data and
        self.generatedImageLabel.configure(image = img2)
        self.generatedImageLabel.image = img2

        #save
        if (random.randint(0, 100) == 1):
            pass
            #self.numberToImageNet.saveNetwork()
            self.imageToNumberNet.saveNetwork()
            self.after(500, self.updateImage)
        else:
            self.after(1, self.updateImage)

def main():

    root = tk.Tk()
    DIP(root)
    root.geometry("500x450")
    root.mainloop()


if __name__ == '__main__':

    '''net1 = SimpleNeuralNetwork.SimpleNeuralNetwork([784, 30, 10])
    net = SimpleNeuralNetwork.SimpleNeuralNetwork([10, 300, 784])
    training_data, validation_data, test_data = dataLoader.load_data_wrapper()
    rtr_d = [[0 for z in xrange(2)] for y in xrange(len(test_data))]
    time.sleep(10)
    for i in range(len(test_data)):
        rtr_d[i][0], rtr_d[i][1] = net1.feedForward(training_data[i][0]), training_data[i][0]

    net.train(rtr_d, 10, 3.0, None)'''

    main()
