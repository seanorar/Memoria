import os
from Tkinter import Tk
import tkFileDialog
toplevel = Tk()
toplevel.withdraw()
filename = tkFileDialog.askopenfilename()
if os.path.isfile(filename):
    for line in open(filename,'r'):
        print line,
else: print 'No file chosen'
raw_input('Ready, push Enter')