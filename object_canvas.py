# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:09:09 2021

@author: Maarten

In this script, the SkeletonCanvas class is defined. The SkeletonCanvas class 
regulates all modifications to the skeleton
"""
#%% import packages
import tkinter as tk
from tkinter import ttk

from rename_object import RenameObject
#%%

class ObjectCanvas(ttk.Notebook):
    
    def __init__(self, master, **kwargs):
        super().__init__(**kwargs)
        
        #assign master
        self.master = master
        
        #assign image object
        self.image = self.master.annotation_canvas.image
        
        self.master.master.update_idletasks()
        
        # tab with all objects
        self.objects = tk.Frame(master=self)
        
        #create listbox and scrollbar
        self.list_objects = tk.Listbox(master=self.objects)
        self.scrollbar = tk.Scrollbar(self.objects,
                                      orient="vertical",
                                      width=20)
    
        #configure scrollbar
        self.scrollbar.config(command=self.list_objects.yview)
        self.list_objects.config(yscrollcommand=self.scrollbar.set)
        
        #bind events to self.list_objects   
        self.list_objects.bind("<<ListboxSelect>>",
                               lambda event: self.activate_object())
        self.list_objects.bind("<Delete>",
                               lambda event: self.delete_object())
        self.list_objects.bind("<Double-1>",
                               lambda event: self.rename_object())
        #<Double-1> = double click of left mouse button
        
        #create button_frame
        self.object_buttons_frm = tk.Frame(master=self.objects)
        self.btn_draw_new_object = tk.Button(master=self.object_buttons_frm,
                                             text="draw new",
                                             command=self.draw_new_object)
        self.btn_delete_object = tk.Button(master=self.object_buttons_frm,
                                           text='delete',
                                           command=self.delete_button_pressed)        
        
        #tab with all mask objects
        self.mask_objects = tk.Frame(master=self)
        
        #create listbox and scrollbar
        self.list_masks = tk.Listbox(master=self.mask_objects)
        self.scrollbar_masks = tk.Scrollbar(self.mask_objects,
                                      orient="vertical",
                                      width=20)
    
        #configure scrollbar
        self.scrollbar_masks.config(command=self.list_masks.yview)
        self.list_masks.config(yscrollcommand=self.scrollbar_masks.set)
        
        #bind events to self.list_objects   
        
        self.list_masks.bind("<<ListboxSelect>>",
                               lambda event: self.activate_mask())
        self.list_masks.bind("<Delete>",
                               lambda event: self.delete_mask())
        self.list_masks.bind("<Double-1>",
                               lambda event: self.rename_mask())
        #<Double-1> = double click of left mouse button
        
        #create button_frame
        self.masks_buttons_frm = tk.Frame(master=self.mask_objects)
        self.btn_draw_new_mask = tk.Button(master=self.masks_buttons_frm,
                                             text="draw new",
                                             command=self.draw_new_mask)
        self.btn_hide_show_masks = tk.Button(master=self.masks_buttons_frm,
                                             text="hide all",
                                             command=self.hide_show_masks)
        self.btn_delete_mask = tk.Button(master=self.masks_buttons_frm,
                                           text='delete',
                                           command=self.delete_mask) 
        
        #position all elements
        
        #add tabs to notebook
        self.add(self.objects, text="Objects")
        self.add(self.mask_objects, text='masks')
        
        #format self.objects
        self.btn_draw_new_object.grid(row=0,
                                      column=0,
                                      sticky='news')
        self.btn_delete_object.grid(row=0,
                                    column=1,
                                    sticky='news')
        self.object_buttons_frm.columnconfigure(0, weight=1)
        self.object_buttons_frm.columnconfigure(1, weight=1)
        
        
        self.object_buttons_frm.pack(side='bottom',
                               fill="x")        
        self.list_objects.pack(side="left",
                               fill=tk.BOTH,
                               expand=True)        
        self.scrollbar.pack(side="left", 
                            fill="y")
        
        #format self.mask_objects
        self.btn_draw_new_mask.grid(row=0,
                                    column=0,
                                    sticky='news')
        self.btn_hide_show_masks.grid(row=0,
                                      column=1,
                                      sticky='news')
        self.btn_delete_mask.grid(row=0,
                                  column=2,
                                  sticky='news')
        self.masks_buttons_frm.columnconfigure(0, weight=1)
        self.masks_buttons_frm.columnconfigure(1, weight=1)
        self.masks_buttons_frm.columnconfigure(2, weight=1)
        
        self.masks_buttons_frm.pack(side='bottom',
                               fill="x")        
        self.list_masks.pack(side="left",
                               fill=tk.BOTH,
                               expand=True)        
        self.scrollbar_masks.pack(side="left", 
                            fill="y")       
        
        #bind method to tab change
        self.bind('<<NotebookTabChanged>>', 
                  lambda event: self.check_mode())
        
        
        #set attributes containing information about the application state        
        self.active_object_index = None
        #index (within listbox) of currently active object
        self.active_mask_index = None
        #index (within listbox) of currently active mask
        self.mode = 0 #0 = object mode; 1 = mask mode
        self.masks_visible = True
        
    def reset(self):
        #delete all current objects
        self.list_objects.delete(0, tk.END)
        
    def add_object(self, obj_name):
        self.list_objects.insert(tk.END, obj_name)
    
    def load_objects(self):  
        #delete all current objects
        self.list_objects.delete(0, tk.END)
        
        #load new objects
        names = self.master.annotation_canvas.names
        for i in names.index:
            self.list_objects.insert(tk.END, names.loc[i,"name"])
    
    def delete_object(self):
        #this method may only be invoked if object_canvas is active from the
        #perspective of the annotation_canvas
        if not self.master.annotation_canvas.object_canvas_active:
            return
        
        if len(self.list_objects.curselection()) == 0:
            #if no objects are selected (or declared), no object may be deleted
            return
        
        obj_name = self.list_objects.get(self.active_object_index)
        self.list_objects.delete(self.list_objects.curselection()[0])
        self.master.annotation_canvas.delete_object(obj_name=obj_name)
    
    def delete_button_pressed(self):
        #activate object_canvas is active from the perspective of annotation_canvas        
        self.master.annotation_canvas.object_canvas_active = True
        
        #if the button was invoked after an object was activated by selction of
        #some keypoint in the annotation canvas, firstly the method activate_object
        #should be executed.
        #if this is not the case, no changes will happen as a result of invoking
        #activate_object()
        self.activate_object()
        
        #delete object
        self.delete_object()
        
        #de-activate object_canvas is active from the perspective of annotation_canvas        
        self.master.annotation_canvas.object_canvas_active = False
    
    def activate_object(self, list_index=None):      
        #change active object
        if self.mode == 1:
            #this method should do nothing when in mask mode
            return
        
        #set attribute, so annotation_canvas knows the object_canvas is currently
        #used
        self.master.annotation_canvas.object_canvas_active = True
        
        if list_index is None:
            current_selection = self.list_objects.curselection()
            if len(current_selection) > 0:
                self.active_object_index = current_selection[0]
        else:
            self.active_object_index = list_index       
               
        if self.active_object_index is not None:
            obj_name = self.list_objects.get(self.active_object_index)
            self.master.annotation_canvas.update_active_object(obj_name=obj_name) 
    
    def rename_object(self):
        #this method is integrated in the software architecture, but disabled,
        #since the data-formats doesn't allow to store names for objects
        pass
        
    def draw_new_object(self):
        self.active_object_index = None
        self.list_objects.select_clear(0, tk.END)
        self.master.annotation_canvas.new_object()
        
    def load_masks(self):  
        #delete all current objects
        self.list_masks.delete(0, tk.END)
        
        #load new masks
        names = self.master.annotation_canvas.names_masks
        for i in names.index:
            self.list_masks.insert(tk.END, names.loc[i,"name"])
        
    def activate_mask(self, list_index=None):        
        if self.mode == 0:
            #this method should do nothing when in object mode
            return
    
        if not self.masks_visible:
            #if the masks are not visible, no mask may be activated
            return
        
        #assign to annotation_canvas no point is active any more
        self.master.annotation_canvas.point_mask_active = False
        
        #assign to annotation_canvas the object_canvas is active
        self.master.annotation_canvas.object_canvas_active = True
        
        #change active mask
        if list_index is None:
            current_selection = self.list_masks.curselection()
            if len(current_selection) > 0:
                self.active_mask_index = current_selection[0]
        else:
            self.active_mask_index = list_index       
               
        if self.active_mask_index is not None:
            mask_name = self.list_masks.get(self.active_mask_index)
            
            if type(mask_name) is tuple:
                mask_name = mask_name[0]
                
            self.master.annotation_canvas.update_active_mask(mask_name=mask_name) 
    
    def delete_mask(self):        
        #this method may only be invoked if object_canvas is active from the
        #perspective of the annotation_canvas
        if not self.master.annotation_canvas.object_canvas_active:
            return
        
        if not self.masks_visible:
            #if the masks are not visible, no mask may be deleted
            return
        
        if len(self.list_masks.curselection()) == 0:
            #if no masks are selected (or declared), no mask can be deleted
            return
        
        mask_index = self.list_masks.curselection()[0]
        mask_name=self.list_masks.get(mask_index)[0]
        self.list_masks.delete(mask_index)
        self.master.annotation_canvas.delete_mask(mask_name=mask_name)
    
    def rename_mask(self):
        
        if not self.masks_visible:
            #if the masks are not visible, no mask may be renamed
            return
        
        #get current name of mask
        self.active_mask_index = self.list_masks.curselection()[0]
        mask_current_name=self.list_masks.get(self.active_mask_index)
        
        if type(mask_current_name) is tuple:
            mask_current_name = mask_current_name[0]       
        
        #get new name of mask
        RenameObject(self, mask_current_name)
        
        #activate renamed mask
        self.activate_mask(list_index=self.active_mask_index)
            
    def draw_new_mask(self):
        
        if not self.masks_visible:
            #if the masks are not visible, no new mask may be drawn
            return
        
        self.active_mask_index = None
        self.master.annotation_canvas.mask_mode = True
        self.master.annotation_canvas.new_object()
    
    def hide_show_masks(self):
        self.master.annotation_canvas.show_mask =\
            not self.master.annotation_canvas.show_mask
            
        if self.master.annotation_canvas.show_mask:
            self.masks_visible = True
            self.btn_hide_show_masks.configure(text="hide all")
            #re-activate buttons and listboxes
            self.list_masks.configure(state = tk.NORMAL)
            self.btn_draw_new_mask.configure(state = tk.ACTIVE)
            self.btn_delete_mask.configure(state = tk.ACTIVE)
        else:
            self.masks_visible = False
            self.btn_hide_show_masks.configure(text="show all")
            #disable buttons and listboxes so no wrong things can happen
            self.list_masks.configure(state = tk.DISABLED)
            self.btn_draw_new_mask.configure(state = tk.DISABLED)
            self.btn_delete_mask.configure(state = tk.DISABLED)
        
        self.master.annotation_canvas.update_image()
    
    def add_mask(self, obj_name):
        self.list_masks.insert(tk.END, obj_name)
        
    def check_mode(self):
        self.mode = self.index(self.select())
        
        self.master.annotation_canvas.new_object()
        
        if self.mode == 0 :
            #mode: making annotations
            self.master.annotation_canvas.mask_mode = False
          
        else:
            #mode: creating masks
            self.master.annotation_canvas.mask_mode = True
