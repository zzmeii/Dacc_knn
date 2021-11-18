from tkinter import *
from tkinter import filedialog

import numpy

from knn_alg import *


class Interface:
    def __int__(self):
        self.window = Tk()
        self.chose_file = Button(self.window, text="Выберете файл")
        self.chose_file.grid(column=1, row=0)
        self.chose_file.bind(self.press_file_button)
        self.file_path = ''
        self.first_text = Label(self.window, text="")
        self.first_text.grid(column=0, row=0)
        self.start_button = Button(self.window, text='Запуск')
        self.start_button.grid(column=1, row=1)
        self.status = Label(self.window, text="Не запущено")
        self.status.grid(column=0, row=1)

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
            if sep < 0.3:
                test_data.append(Dot(data[i][:-1], data[i][-1]))
            else:
                unk_data.append(Dot(data[i][:-1]))
        result = start_knn(test_data, unk_data, 3)
