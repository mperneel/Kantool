# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:10:19 2022

@author: Maarten

Defenition of the AnnotationSettings class. This class is used to modify the 
settings of the annotation window
"""

#%% packages
import tkinter as tk


#%%

class AnnotationSettings(tk.Toplevel):
    
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.master = master        

        #general child-window settings
        #set child to be on top of the main window
        self.transient(master)
        #hijack all commands from the master (clicks on the main window are ignored)
        self.grab_set()
        #pause anything on the main window until this one closes (optional)
        self.master.wait_window(self)