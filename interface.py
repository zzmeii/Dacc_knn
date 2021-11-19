from tkinter import filedialog, Tk, Button, Label, Spinbox

import numpy
from matplotlib import pyplot as plt

from knn_alg import *


class Interface:
    def __init__(self):
        self.window = Tk()
        self.window.title("Даций KNN")
        self.chose_file = Button(self.window, text="Выберете файл")
        self.chose_file.grid(column=1, row=0)
        self.chose_file.bind('<Button-1>', self.press_file_button)
        self.file_path = ''
        self.first_text = Label(self.window, text="")
        self.first_text.grid(column=0, row=0)
        self.start_button = Button(self.window, text='Запуск')
        self.start_button.grid(column=2, row=1)
        self.status = Label(self.window, text="Не запущено")
        self.status.grid(column=0, row=1)
        self.start_button.bind('<Button-1>', self.press_start)
        self.result = []
        self.show_g = Button(self.window, text="Показать график")
        self.show_g.grid(column=2, row=2)
        self.show_g.bind('<Button-1>', self.make_graph)
        self.nud = Spinbox(self.window, from_=3, to=11, increment=2)
        self.nud.grid(column=1, row=1)

    def press_file_button(self, event):
        self.file_path = filedialog.askopenfilename()
        self.first_text['text'] = self.file_path

    def press_start(self, event):
        if self.file_path == '':
            self.status['text'] = 'Файл не выбран'
            return
        test_data = []
        unk_data = []
        data = numpy.loadtxt(self.file_path, delimiter=',')
        sep = numpy.random.random(len(data))
        for i in range(len(data)):
            if sep[i] < 0.3:
                test_data.append(Dot(data[i][:-1], data[i][-1]))
            else:
                unk_data.append(Dot(data[i][:-1]))
        self.result = convert(start_knn(test_data, unk_data, int(self.nud.get())))
        self.status['text'] = 'Готово'

    def make_graph(self, event):
        plt.scatter(self.result[0], self.result[1], c=self.result[2])
        plt.show()


if __name__ == '__main__':
    temp = Interface()
    # temp.file_path = 'data.csv'
    temp.window.mainloop()
