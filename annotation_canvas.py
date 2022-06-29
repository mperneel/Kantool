# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:05:13 2021

@author: Maarten

In this script, the AnnotationCanvas class is defined. AnnotationCanvas regulates
all modifications to the annotations
"""
#%% import packages
import os
import tkinter as tk
import tkinter.messagebox as tkmessagebox
import tkinter.filedialog as tkfiledialog
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageTk
import json

from general_image_canvas import GeneralImageCanvas

#%%main code

class AnnotationCanvas(GeneralImageCanvas):
    """
    AnnotationCanvas regulates all modifications to the annotations
    """
    def __init__(self, skeleton, **kwargs):
        super().__init__(**kwargs)
        self.currently_saved = True #saved status
        self.point_active = False #is there an active point?

        self.skeleton = skeleton
        self.annotations = None
        self.marginal_keypoint_index = -1
        self.keypoint_index = 0
        self.object = 0
        self.currently_saved = True
        self.object_memory = None
        self.new_keypoint_created = False
        self.current_mask_confirmed = True #is the current mask confirmed?

        self.image_inter_scale = 1.0 #scale of self.image_inter
        self.image = None #image matrix
        self.image_inter = None #intermediary image matrix
        #basic modifications to self.image are done once and then stored
        #in self.image_inter to speed up the code
        self.image_inter_1 = None
        self.image_shown = None #matrix with shown image

        self.keypoint_index_memory = 0
        self.keypoint_reactivated = False
        self.point_mask_reactivated = False
        
        self.mask_mode = False #mask mode
        #if False, objects are annotated, if True, masks wil be drawn
        self.show_mask = True #True if masks should be drawn
        self.linewidth = 1 #width of the edges of the active mask
        self.mask_id = 0 #id of the currently active mask
        
        #names of objects
        self.names = pd.DataFrame(data=None,
                                  columns=["obj", "name"])
        
        #points of current masks
        self.points_mask = np.empty(shape=(0,2))
        #masks are defined as polygons
        
        #all points of masks
        self.annotation_points_masks = pd.DataFrame(data=None,
                                               columns=["obj", "x", "y"])
        
        #names of masks
        self.names_masks = pd.DataFrame(data=None,
                                        columns=["obj", "name"])
        
        self.object_canvas_active = False

        self.bind("<Button-1>",
                  self.button_1)
        self.bind("<ButtonRelease-1>",
                  self.button_1_release)
        self.bind("<Button-3>",
                  lambda event: self.button_3())
        self.bind("<B1-Motion>",
                  self.motion_b1)
        self.bind("<Motion>",
                  self.motion)

    def update_image(self, mode=0):
        """
        Update image

        mode : int
            0 : the image is constructed from scratch\n
            1 : the code starts from the rescaled image with all non-active
            objects drawn. This mode is used to draw the currently active object/mask
            and the mouse circle\n
            2 : the code starts from the rescaled image with all objects/masks drawn.
            This mode is used to draw the mouse circle
            3: the code starts from scratch. The active object is emphasized by
            drawing a circle around it's keypoints

        """

        if self.image_name is None:
            #no image is loaded
            #display nothing
            self.itemconfigure(self.image_photoimage, image=None)
            self._image_photoimage = None
            return

        #There is an image loaded

        #get image height and width
        image_height, image_width = self.image.shape[:2]

        #get scale and intercepts
        s = self.zoom_level
        dx = self.zoom_delta_x
        dy = self.zoom_delta_y

        #get annotations
        df = self.annotations

        #rescale image
        if (mode in [0, 3]) or (self.image_inter_scale != s):
            #resize self.image_inter according to scale s
            self.image_inter = cv2.resize(self.image,
                                          dsize=(int(image_width * s),
                                                 int(image_height * s)))
            self.image_inter_scale = s

        #draw skeletons of non-active objects
        if (mode in [0, 3]) or (self.image_inter_scale != s):
            for i in df.index:
                if i != self.object:
                    data_object = df.loc[i, :].to_numpy()
                    for j, _ in enumerate(self.skeleton.keypoints):
                        if not np.isnan(data_object[2 * j]):
                            x, y = (data_object[2 * j: 2 * j + 2] * s).astype(int)

                            #draw lines
                            parent_id = self.skeleton.parent[j]
                            if (parent_id != -1) and\
                                (not np.isnan(data_object[2 * parent_id])):
                                end_point = (x, y)
                                start_point = data_object[2 * parent_id:
                                                          2 * parent_id + 2]
                                start_point = tuple((start_point * s).astype(int))
                                color = self.skeleton.color[j]
                                cv2.line(self.image_inter,
                                         start_point,
                                         end_point,
                                         color=color,
                                         thickness=3)

                            #draw points
                            color = self.skeleton.color[j]
                            cv2.circle(self.image_inter,
                                       (x, y),
                                       radius=5,
                                       color=color,
                                       thickness=-1)
                        
            #draw all non-active masks
            if self.show_mask:
                for i in self.names_masks["obj"]:
                    if i != self.mask_id:
                        mask = self.annotation_points_masks.loc[self.annotation_points_masks["obj"]==i,\
                                                             ["x", "y"]].to_numpy()
                        color = [0, 0, 0]
                        
                        self.image_inter = self.draw_mask(self.image_inter, mask,\
                                                          s, color=color)    

        #draw skeleton of active object        
        
        if mode in [0, 1, 3]:
            self.image_inter_1 = self.image_inter.copy()
            
            i = self.object
            data_object = df.loc[i, :].to_numpy()
            for j, _ in enumerate(self.skeleton.keypoints):
                if not np.isnan(data_object[2 * j]):
                    x, y = (data_object[2 * j: 2 * j + 2] * s).astype(int)

                    #draw lines
                    parent_id = self.skeleton.parent[j]
                    if (parent_id != -1) and\
                        (not np.isnan(data_object[2 * parent_id])):
                        end_point = (x, y)
                        start_point = data_object[2 * parent_id:
                                                  2 * parent_id + 2]
                        start_point = tuple((start_point * s).astype(int))
                        color = self.skeleton.color[j]
                        cv2.line(self.image_inter_1,
                                 start_point,
                                 end_point,
                                 color=color,
                                 thickness=3)

                    #draw points
                    color = self.skeleton.color[j]
                    cv2.circle(self.image_inter_1, (x, y),
                               radius=5,
                               color=color,
                               thickness=-1)
                    
            #draw points of active mask
            if self.show_mask:
                for point in self.points_mask:
                    x = int(point[0] * s)
                    y = int(point[1] * s)
                    self.image_inter_1= cv2.circle(self.image_inter_1,
                                                 (x,y),
                                                 radius=3,
                                                 color=[0,255,0],
                                                 thickness=-1)
    
                color = [255, 255, 0] #yellow
                mask = self.points_mask.copy()
                
                if len(mask)>=3:
                    #draw segment
                    #identical to drawing segments of active object
                    mask_color = color
                    self.image_inter_1 = self.draw_segment(self.image_inter_1, mask,\
                                                         s, color=mask_color)

        #draw circle around mouse/activated keypoint
        
        if mode in [0,1,2]:
            self.image_shown = self.image_inter_1.copy()
            
            if not self.mask_mode:
    
                if self.keypoint_reactivated:
                    #draw circle around activated keypoint
                    x, y = (df.iloc[self.object,
                                    self.keypoint_index * 2: self.keypoint_index * 2 + 2]
                            * s).astype(int)
                else: #not self.keypoint_reactivated:
                    #draw circle around mouse
                    x = int(self.mouse_x  + self.zoom_delta_x)
                    y = int(self.mouse_y  + self.zoom_delta_y)
        
                color = self.skeleton.color[self.keypoint_index]
                cv2.circle(self.image_shown, (x, y),
                           radius=20,
                           color=color,
                           thickness=3)
            else: #self.mask_mode:
                #if in mask mode, nothing should be drawn around the mouse
                pass
            
        #draw circles around keypoints of re-activated object
        
        if mode == 3:
            self.image_shown = self.image_inter_1.copy()
            
            if not self.mask_mode:
                i = self.object
                data_object = df.loc[i, :].to_numpy()
                for j, _ in enumerate(self.skeleton.keypoints):
                    if not np.isnan(data_object[2 * j]):
                        x, y = (data_object[2 * j: 2 * j + 2] * s).astype(int)

                        #draw circles around points
                        color = self.skeleton.color[j]
                        cv2.circle(self.image_shown, (x, y),
                                   radius=20,
                                   color=color,
                                   thickness=3)
            else: #self.mask_mode
                #if in mask mode, nothing should be drawn around the mouse
                pass

        #slice self.image_shown so slice fits in self
        uw = self.winfo_width()
        uh = self.winfo_height()
        if (self.image_shown.shape[1] > uw) or\
            (self.image_shown.shape[0] > uh):
            self.image_shown = self.image_shown[dy : dy + uh,
                                                dx: dx + uw,
                                                :]

        #show image
        image_shown = Image.fromarray(self.image_shown)
        image_shown = ImageTk.PhotoImage(image_shown)

        self.itemconfigure(self.image_photoimage, image=image_shown)
        self._image_photoimage = image_shown
        
    def draw_segment(self, image, segment, z, color):
        """
        Draw segment on image

        Segments contains all the points which define the segment

        z is the scale of the image relative to the original image
        """
        #rescale coordinates within segment
        segment *= z

        #round coordinates and draw segment on image
        segment = np.int64(segment).reshape((-1, 1, 2))

        image= cv2.polylines(image,
                            [segment],
                            isClosed=True,
                            color=color,
                            thickness=self.linewidth)
        return image
        
    def draw_mask(self, image, mask, z, color):
        """
        Draw mask on image

        a mask is a polygon defined by a list of points
        """
        #rescale coordinates within mask
        mask *= z

        #round coordinates and draw mask on image
        mask = np.int64(mask).reshape((-1, 1, 2))

        image= cv2.fillPoly(image,
                            [mask],
                            color=color)
        
        return image        

    def open_image(self):
        """
        Open a dialog to choose an image to load
        """
        #check if a project is opened
        if self.wdir is None:
            return

        #ask for a filepath
        filepath = tkfiledialog.askopenfilename()

        if filepath != "":
            #load the image (and annotations)
            self.load_image(filepath, full_path=True)

    def load_image(self, filename, full_path=True):
        """
        Decisive method to load an image, together with it's annotations (if
        availabe)

        Parameters
        ----------
        filename : string
            DESCRIPTION.
        full_path : bool, optional
            DESCRIPTION. The default is True.
        """

        #set wdir and image_name
        if full_path is True:
            wdir, image_name = os.path.split(filename)
            #check if image is in project folder
            if wdir != self.wdir:
                message = "The file you selected is not located in the project folder"
                tkmessagebox.showerror(title='Invalid file',
                                       message=message)
                return
        else: #full_path = False
            wdir = self.wdir
            image_name = filename

        #check if file is a valid image
        if (len(image_name.split(".")) > 1) and \
            image_name.split(".")[1] in ['jpg', 'png']:

            if self.currently_saved:
                #annotations are saved (or there is currenlty no image shown)
                self.image_name = image_name
                self.import_image()
            else:
                #there are currently unsaved changes in the annotations
                message = "do you want to save the annotation changes you made " +\
                          "for the current image?"
                answer = tkmessagebox.askyesnocancel("Save modifications",
                                                     message=message)
                if answer is True:
                    #Save changes
                    self.save()
                    self.image_name = image_name
                    self.import_image()
                elif answer is False:
                    #Discard changes
                    self.image_name = image_name
                    self.import_image()
                #else: #answer==None
                    #nothing has to be done
        else:
            #there was selected an object, but it was not a supported image
            tkmessagebox.showerror("Invalid file",
                                   "Only files with the extension .jpg or .png are supported")

    def import_image(self):
        """
        Executive method to import image and load annotations (if available)
        """
        #reset parameters
        self.reset_parameters()
        self.reset_masks()

        #Update title of application
        self.master.master.title("CoBRA annotation tool " + self.image_name)

        #import image
        image = cv2.imread(self.image_name)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #import annotations (if they exist)
        if self.image_name.split('.')[0] + '.csv' in os.listdir():
            #create list of column names
            columns = []
            for name in self.skeleton.keypoints:
                columns.append(name + "_x")
                columns.append(name + "_y")

            #load annotations
            try:
                self.annotations = pd.read_csv(self.image_name.split('.')[0] + '.csv',
                                               sep=",",
                                               header=None)
                self.annotations.columns = columns
            except pd.errors.EmptyDataError:
                #create epty dataframe
                self.annotations = pd.DataFrame(columns=columns,
                                                dtype=float)
                self.annotations = self.annotations.append(pd.Series(dtype='float64'),
                                                           ignore_index=True)
              
            #create generic names for objects
            names_array = np.arange(len(self.annotations.index))
            names_dict = {"obj": names_array.copy(),
                          "name": [str(i) for i in names_array]}
            self.names = pd.DataFrame(names_dict)
        
        #load names of objects in object_canvas
        self.master.object_canvas.load_objects()
        
        #import masks (if they exist)
        self.load_masks()
        
        #load names of masks in object_canvas
        self.master.object_canvas.load_masks()

        #Set initial zoom level
        self.reset_zoom_level()

        #Update image
        self.new_object()
        self.update_image(mode=0)
        
    def load_masks(self):
        """
        load masks
        """
        
        #look if there is a .json file with masks
        if self.image_name.split(".")[0] + "_mask.json" in os.listdir():
            #read json file
            json_file = open(self.image_name.split(".")[0] + "_mask.json")
            masks = json.load(json_file)
            json_file.close()
        
            #write data in masks to right attribute dataframes of self
            self.mask_id = 0
            for i in masks.keys():
                mask = masks[i]
                
                #add name to self.names                
                df = self.names_masks
                if 'name' in mask:
                    new_mask = pd.DataFrame(data=[[self.mask_id, mask["name"]]],
                                          columns=df.columns)
                else:
                    #this case is present to keep compatability with annotations
                    #made with a previous version of CoBRA annotation tool
                    new_mask = pd.DataFrame(data=[[self.mask_id, i]],
                                          columns=df.columns)
                df = pd.concat([df, new_mask],
                               ignore_index=True)
                self.names_masks = df                
        
                #add points to self.annotation_points_masks
                points_dict = mask["points"]
                points = []
                for j in points_dict.keys():
                    x = points_dict[j]["x"]
                    y = points_dict[j]["y"]
                    points.append([x, y])
                points = pd.DataFrame(data=points,
                                  columns=["x", "y"])
                points["obj"] = self.mask_id
                self.annotation_points_masks = \
                    pd.concat([self.annotation_points_masks, points],
                              ignore_index=True)
        
                #increase self.mask_id with 1
                self.mask_id += 1
        else:
            #There is no .json file with masks available
            #reset all mask related attributes to their default value
            self.reset_masks()

    def reset_parameters(self):
        """
        Reset attributes to their default value
        """
        self.zoom_level = 1.0
        self.zoom_delta_x = 0
        self.zoom_delta_y = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.image_inter_scale = 1.0

        self.image = None
        self.image_inter = None
        self.image_inter_1 = None
        self.image_shown = None

        self.object = 0
        self.keypoint_index = 0
        self.marginal_keypoint_index = -1

        self.currently_saved = True
        self.keypoint_reactivated = False

        #reset annotations
        if len(self.skeleton.keypoints) > 0:
            columns = []
            for name in self.skeleton.keypoints:
                columns.append(name + "_x")
                columns.append(name + "_y")

            self.annotations = pd.DataFrame(columns=columns,
                                            index=[0],
                                            dtype=float)
        else:
            self.annotations = None
            
        #reset names
        self.names = pd.DataFrame(data=None,
                                  columns=["obj", "name"])

        self.image_photoimage = self.create_image(0, 0, anchor='nw')
        
    def reset_masks(self):
        """
        reset all parameters related to the masks
        """
        self.mask_id = 0 #id current mask
        self.points_mask = np.empty(shape=(0,2)) #current mask
        self.annotation_points_masks = pd.DataFrame(data=None,
                                               columns=["obj", "x", "y"])
        self.names_masks = pd.DataFrame(data=None,
                                        columns=["obj", "name"])

    def switch_image(self, direction=1):
        """
        Load the next/previous image in the working directory
        """

        if self.wdir != None and self.image_name != None:
            #Check if there is an image loaded

            #list all files within working directory
            files = np.array(os.listdir(self.wdir))

            #get index of current image
            file_number = np.argmax(files == self.image_name)

            #look for next/previous image
            keep_looking = True
            while keep_looking is True:
                if file_number == len(files) - 1:
                    if direction == 1:
                        file_number = 0
                    else:
                        file_number += direction
                elif file_number == 0:
                    if direction == -1:
                        file_number = len(files) - 1
                    else:
                        file_number += direction
                else:
                    file_number += direction

                image_name = files[file_number]

                #check if file extension is correct
                if (len(image_name.split(".")) > 1) and\
                    (image_name.split(".")[1] in ['jpg', 'png']):
                    keep_looking = False
                    #remark: if current image is the only image in the
                    #folder, the while loop will be ended once we
                    #re-encounter the name of the current image

            #load image
            self.load_image(image_name, full_path=False)

    def save(self):
        """
        Save the annotations
        """
        #check if an image is loaded
        if self.image is None:
            return

        #remove NaN-objects
        self.remove_nan_objects()

        #save annotations
        filename = self.image_name.split(".")[0]
        self.annotations.to_csv(filename + '.csv',
                                header=False,
                                index=False,
                                float_format='%.2f')
        
        #save masks
        self.save_masks()

        #set state of self.currently_saved to True
        self.currently_saved = True
        
    def save_masks(self):
        """
        Save masks

        The masks will be saved in the directory of the image in a .json
        file with following filename: imagename_mask.json
        """
        
        if len(self.points_mask)>0:
            #if it's possible to save the current (non-confirmed) mask,
            #save this mask
            self.new_mask()

        #all information that has to be saved is in self.annotation_points_masks and
        #self.names_masks

        if len(self.annotation_points_masks)==0:
            #there is no data to be saved or all data was removed

            #check if there exists a .json file for the opened image and
            #remove it if existing (empty .json files are unwanted)
            if self.image_name.split(".")[0] + "_mask.json" in os.listdir(self.project.wdir):
                os.remove(self.image_name.split(".")[0] + "_mask.json")
        else:
            #there is data to save
            #fuse self.names_masks, self.annotation_points_masks in one dictionary
            data={}
            for mask_id in self.names_masks["obj"]:
                #get mask_name
                mask_name = self.names_masks.loc[self.names_masks["obj"]==mask_id, 'name'].iloc[0]
                
                #get mask points
                points = self.annotation_points_masks.loc[self.annotation_points_masks["obj"]==mask_id,\
                                                    ["x","y"]].round(2)

                #convert points from pd.DataFrame to dict
                points_dict = {}
                for i in range(len(points)):
                    points_dict[i] = points.iloc[i,].to_dict(({}))

                #assembly dict for object
                mask = {"name": mask_name,
                       "points": points_dict}

                #add object dict to data
                data[mask_id] = mask

            #save data to .json file
            file = open(self.image_name[:-4] + "_mask.json", 'w')
            json.dump(data, file, indent = 3)
            file.close()

    def new_point(self, event):
        #decisive method to distinguish between mask mode and annotation mode
        if self.mask_mode:
            self.new_mask_point(event)
        else:
            self.new_keypoint(event)
            
    def new_keypoint(self, event):
        """
        Create a new keypoint

        Parameters
        ----------
        event : tkinter.Event
            ButtonPress event at the position where a new keypoint should be created
        """
        df = self.annotations
        s = self.zoom_level

        df.iloc[self.object, self.keypoint_index * 2] = (event.x + self.zoom_delta_x) /\
            (s)
        df.iloc[self.object, self.keypoint_index * 2 + 1] = (event.y + self.zoom_delta_y) /\
            (s)

        self.annotations = df
        self.new_keypoint_created = True

        #set state of self.currently_saved to False
        self.currently_saved = False

        #update image
        self.update_image(mode=1)
        
    def new_mask_point(self, event):
        #add a new point to the mask
        
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return
        
        #get scale of image        
        s = self.zoom_level
        
        #get x and y coordinate of point
        x = (event.x + self.zoom_delta_x) / s
        y = (event.y + self.zoom_delta_y) / s        
        
        #set saved state to false
        self.currently_saved = False
        
        #set object_confirmed state to False
        self.current_mask_confirmed = False        
        
        self.point_active = False
        if (x<=self.image.shape[1]) and (y<=self.image.shape[0]):
            #Check if a point in the image is activated and not a point
            #in the surrounding grey area
            
            #if a point is close to the borders, adapt x and y, so point is
            #drawn on the borders
            if x * s <= 40:
                x = 0
            elif (self.image.shape[1] - x) * s <= 40:
                x = self.image.shape[1] - 1
                
            if y * s <= 40:
                y = 0
            elif (self.image.shape[0] - y) * s <= 40:
                y = self.image.shape[0] - 1            
            
            self.point_mask_active = True
            if len(self.points_mask)>0:
                #check if you have to replace a yet existing point or if you have
                #to create a new point
                dist_to_points = np.sqrt(np.sum(np.square(\
                            self.points_mask - np.array([[x, y]])), axis=1).\
                                         astype(float)) * s
                
                if min(dist_to_points)<=10/s:
                    #replace a yet existing point
                    self.point_id = np.argmin(dist_to_points)
                else:
                    #add an extra point
                    if len(self.points_mask) >=3:
                        #check if you have to add an extra point at the end, or you have to
                        #split an edge

                        #calculate orthogonal distance to edges
                        C = np.array([[x * s, y * s]])
                        points = self.points_mask * s

                        A = points
                        B = np.concatenate((points[1:], [points[0]]), axis=0)
                        u = B-A
                        v = np.concatenate((-u[:,1].reshape(-1,1), u[:,0].\
                                            reshape(-1,1)), axis=1)
                        v = v / np.linalg.norm(v, axis=1).reshape(-1,1)

                        ac = A - C #vectors from A to C

                        #orthogonal distances to edges
                        d = np.abs((ac * v).sum(-1))

                        if min(d)<=10/s:
                            #split (possibly) an edge
                            insert_index = np.argmin(d) + 1

                            #check if point is in sensitive zone
                            i = insert_index
                            A = self.points_mask.copy()
                            A = np.concatenate((A, A[0,:].reshape(-1,2)), axis=0)
                            x_min = min(A[i - 1,0], A[i,0])
                            x_max = max(A[i - 1,0], A[i,0])
                            y_min = min(A[i - 1,1], A[i,1])
                            y_max = max(A[i - 1,1], A[i,1])
                            if (x<=x_max) and (x>=x_min) and\
                                (y<=y_max) and (y>=y_min):
                                #point is in sensitive zone
                                #split edge
                                self.points_mask = np.concatenate(\
                                    (self.points_mask[:insert_index].reshape(-1,2),
                                     [[x, y]],
                                     self.points_mask[insert_index:].reshape(-1,2)),
                                    axis=0)
                                self.point_id = insert_index
                            else:
                                #point is not in sensitive zone
                                #add extra point at the end of self.points_mask
                                self.point_id = len(self.points_mask)
                                self.points_mask = np.append(self.points_mask,[[x, y]], axis=0)
                        else: #min(d)>dist_max
                            #add extra point at the end of self.points
                            self.point_id = len(self.points_mask)
                            self.points_mask = np.append(self.points_mask,[[x, y]], axis=0)
                    else: #len(self.points)<3:
                        #add extra point at the end of self.points
                        self.point_id = len(self.points_mask)
                        self.points_mask = np.append(self.points_mask,[[x, y]], axis=0)
            else: #len(self.points)==0:
                #add first point in self.points
                self.point_id = len(self.points_mask)
                self.points_mask = np.append(self.points_mask,[[x, y]], axis=0)

            #if a valid point was added, return True
            return True
        
        #if no valid point was added, return False
        return False
        pass

    def skip_keypoint(self):
        """
        Skip the current keypoint and go to the next keypoint of the skeleton
        """
        #if we enter this method, there is always at least one keypoint missing
        df = self.annotations

        missing_keypoint_found = False

        while not missing_keypoint_found:
            self.marginal_keypoint_index = (self.marginal_keypoint_index + 1) %\
                len(self.skeleton.keypoints)
            self.keypoint_index = self.skeleton.func_annotation_order[self.marginal_keypoint_index]
            if np.isnan(self.annotations.iloc[self.object, 2*self.keypoint_index]):
                missing_keypoint_found = True

        self.annotations = df

        self.update_image(mode=2)

    def keypoint_searching(self, event):
        """
        Update search circle around the mouse

        Parameters
        ----------
        event : tkinter.Event
            Motion event containing the position of the mouse
        """

        self.mouse_x = event.x
        self.mouse_y = event.y
        if self.image is not None:
            self.update_image(mode=2)

    def motion_b1(self, event):
        """
        Desicive method to move a keypoint

        Parameters
        ----------
        event : tkinter.Event
            Motion event
        """
        if self.keypoint_reactivated or self.new_keypoint_created:
            self.move_current_keypoint(event)
        elif self.mask_mode:
            self.update_point_mask(event)

    def move_current_keypoint(self, event):
        """
        Executive method to move a keypoint

        Parameters
        ----------
        event : tkinter.Event
            Motion event
        """
        i = self.keypoint_index

        s = self.zoom_level

        #update keypoint coordinates
        df = self.annotations
        df.iloc[self.object, i * 2] = (event.x + self.zoom_delta_x) / s
        df.iloc[self.object, i * 2 + 1] = (event.y + self.zoom_delta_y) /s
        self.annotations = df

        #set state of currently_saved to False
        self.currently_saved = False

        #update mouse positions
        self.mouse_x = event.x
        self.mouse_y = event.y

        #update the shown image
        self.update_image(mode=1)
        
    def update_point_mask(self, event):
         
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return
        
        #set saved state to false
        self.currently_saved = False
        
        #set object_confirmed state to False
        self.current_mask_confirmed = False
        
        #get scale
        s = self.zoom_level
        
        #get x and y
        x = (event.x + self.zoom_delta_x) / s
        y = (event.y + self.zoom_delta_y) /s
        
        #update position of currently active point
        self.points_mask[self.point_id,:] = np.array([x, y])
        
        #update the shown image
        self.update_image(mode=1)

    def activate_next_keypoint_searching(self):
        """
        Update all attributes so the next missing keypoint can be looked for.
        If an object is complete, a new object will be created
        """
        missing_keypoint_found = False
        n_keypoints = len(self.skeleton.keypoints)
        loop = 0

        while not missing_keypoint_found:
            loop += 1

            self.marginal_keypoint_index = (self.marginal_keypoint_index + 1) %\
                len(self.skeleton.keypoints)
            self.keypoint_index = self.skeleton.func_annotation_order[self.marginal_keypoint_index]
            if np.isnan(self.annotations.iloc[self.object, 2*self.keypoint_index]):
                missing_keypoint_found = True
            elif loop == n_keypoints:
                #confirm current object and create new object
                self.new_object()
                return

        if self.keypoint_reactivated:
            self.keypoint_index = self.keypoint_index_memory
            self.object = self.object_memory

        self.update_image(mode=0)

    def button_1(self, event):
        """
        Decisive method to re-activate a keypoint or draw a new keypoint
        """
        #check if an image was loaded
        if self.image_name is None:
            return

        #check if no point is re-activated
        if not self.keypoint_reactivated:

            s = self.zoom_level

            x = (event.x  + self.zoom_delta_x) / s
            y = (event.y  + self.zoom_delta_y) / s

            location = np.array([x, y])

            if len(self.annotations) > 0:
                for i in range(len(self.skeleton.keypoints)):
                    coordinates = self.annotations.iloc[:, 2*i:2*i+2].to_numpy()
                    distance = np.sqrt(np.sum((coordinates - location)**2, axis=1))
                    if (sum(np.isnan(distance)) < len(distance)) and\
                        (np.nanmin(distance) < 10/s):
                        self.keypoint_reactivated = True

                        self.keypoint_index_memory = self.keypoint_index
                        self.object_memory = self.object
                        self.keypoint_index = i
                        self.marginal_keypoint_index = \
                            np.where(self.keypoint_index == \
                                     np.array(self.skeleton.func_annotation_order))[0][0]
                        #np.where() returns a tuple containing a numpy array
                        #with the indices for which the condition is True, we
                        #need the first element of that matrix (which is the
                        #first and only element of the tuple)
                        self.object = np.nanargmin(distance)
                        
                        #update object_canvas
                        self.master.object_canvas.list_objects.select_clear(0, tk.END)
                        #blue color
                        self.master.object_canvas.list_objects.select_set(self.object)
                        #underlining
                        self.master.object_canvas.list_objects.activate(self.object)
                        
                        self.update_image(mode=0)

            if not self.keypoint_reactivated:
                self.new_point(event)

        else: #keypoint_reactivated:
            self.keypoint_reactivated = False

    def button_3(self):
        """
        Decisive method to skip the current keypoint where we are looking for or
        to de-activate the currently activated keypoint
        """
        #check if an image was loaded
        if self.image_name is None:
            return

        if not self.keypoint_reactivated:
            self.skip_keypoint()
        else: #self.keypoint_reactivated:
            self.keypoint_reactivated = False
            self.activate_next_keypoint_searching()
            self.update_image(mode=2)

    def new_object(self):
        """
        Create a new object
        """
        if len(self.skeleton.keypoints) == 0 :
            #if len(self.skeleton.keypoints) == 0, this means there is currently
            #no project loaded
            
            #therefore, we return immediately
            return
        
        if self.mask_mode:
            #if we are in mask mode, a new mask should be initiated
            self.new_mask()
            return
        
        self.marginal_keypoint_index = 0
        self.keypoint_index = self.skeleton.func_annotation_order[self.marginal_keypoint_index]
        
        #remove all objects with only NaN coordinates
        self.remove_nan_objects()
        
        #load all names still present in object_canvas
        self.master.object_canvas.load_objects()

        #Add new row for new object
        df = self.annotations
        new_row = pd.DataFrame(columns=df.columns,
                               index=[0],
                               dtype=float)
        df = pd.concat([df, new_row], ignore_index=True)
        self.annotations = df
        
        self.object = self.annotations.shape[0] - 1
        #Python is zero based, so to get the index of the last object,
        #we call the number of objects and substract one
        
        #add new generic name to self.names
        if len(self.names) > 0:
            new_name = str(int(self.names["name"].iloc[-1]) + 1)
            new_row = pd.DataFrame(data=[[self.object, new_name]],
                                   columns=["obj", "name"])
            self.names = pd.concat([self.names, new_row],
                                   ignore_index=True)
        else: #len(self.names) == 0:
            self.names = pd.DataFrame(data=[[0, '0']],
                                      columns=['obj', 'name'])
            
        #update state booleans
        self.keypoint_reactivated = False
        
        #updat image
        self.update_image(mode=0)
        
    def new_mask(self):
        """
        Confirm current mask
        """
        
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return

        if len(self.points_mask)>=3:
            #a mask can be only confirmed if it has at least 3 points

            #save class
            if self.mask_id in self.names_masks["obj"]:
                #adapted mask
                
                #update self.annotation_points_masks
                indices_to_drop = self.annotation_points_masks.loc[self.annotation_points_masks["obj"] == self.mask_id,].index
                self.annotation_points_masks = self.annotation_points_masks.drop(indices_to_drop)                
                
                points = pd.DataFrame(data=self.points_mask,
                                      columns=["x", "y"])
                points["obj"] = self.mask_id
                
                self.annotation_points_masks = \
                    pd.concat([self.annotation_points_masks, points],
                              ignore_index=True)
                self.points_mask = np.empty(shape=(0,2))
                
            else:
                #new mask
                
                #save name of mask (default)
                dict_name = {'obj': [self.mask_id],
                             'name': [self.mask_id]}
                
                frame_name = pd.DataFrame.from_dict(dict_name)
                
                self.names_masks = pd.concat([self.names_masks, frame_name],
                                             ignore_index=True)           
                
                #add self.points to self.annotation_points
                points = pd.DataFrame(data=self.points_mask,
                                      columns=["x", "y"])
                points["obj"] = self.mask_id
                
                self.annotation_points_masks = \
                    pd.concat([self.annotation_points_masks, points],
                              ignore_index=True)
                self.points_mask = np.empty(shape=(0,2))
                
                #add object to listbox of object_canvas
                self.master.object_canvas.add_mask(dict_name["name"])
      
            #set self.obj_id to object id of next object
            self.mask_id = self.names_masks["obj"].max() + 1
            
            #set object_confirmed state to False
            self.current_mask_confirmed = True

            #update image
            self.update_image(mode=0)
            
    def update_active_mask(self, mask_id=None, mask_name=None):
        
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return
        
        #process arguments
        if mask_id is None and mask_name is None:
            raise ValueError("mask_id and mask_name may not be both None")
        elif mask_id is not None and mask_name is not None:
            raise ValueError("mask_id and mask_name may not be given both")
        elif mask_name is not None:
            #only mask_name is given
            #get mask_id
            mask_id = self.names_masks.loc[self.names_masks["name"]==mask_name, "obj"].iloc[0]
            
            
        #first store all adaptations to the current mask (if necessary) 
        #to the general dataframes
        if self.current_mask_confirmed is False:
            self.new_mask()
        
        #set id of active object
        self.mask_id = mask_id #id of current object
        
        #set points of active object
        self.points_mask = self.annotation_points_masks.loc[self.annotation_points_masks["obj"]==self.mask_id,
                                                 ["x", "y"]].to_numpy()
        
        #set id of last active point of mask (set to last point of mask)
        self.point_id = len(self.points_mask) - 1 
                
        self.update_image(mode=0)

    def remove_nan_objects(self):
        """
        Remove all objects without keypoints
        """
        #get indices of empty objects
        nan_objects = self.annotations.isnull().all(axis=1)
        
        #remove empty objects
        self.annotations = self.annotations.loc[~nan_objects,]  
        self.names = self.names.loc[~nan_objects,:]
        #the tilde inverts the boolean array
        
        #set object id's correctly
        self.names["obj"] = [i for i in range(len(self.names))]

    def delete_keypoint(self):
        """
        Delete the currently activated keypoint
        """
        if self.keypoint_reactivated:
            df = self.annotations
            i = self.keypoint_index
            df.iloc[self.object, i * 2: i * 2 + 2] = np.nan
            self.keypoint_reactivated = False
            #decrease self.marginal_keypoint_index with 1 so self.activate_next_keypoint_searching
            #will 'activate' the search to a keypoint of the type of the deleted
            #keypoint
            self.marginal_keypoint_index -= 1
            self.activate_next_keypoint_searching()

            #set state of self.currently_saved to False
            self.currently_saved = False

            #update image
            self.update_image(mode=1)
            
    def delete_mask(self, mask_id=None, mask_name=None):
        
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return
        
        #process arguments
        if mask_id is None and mask_name is None:
            raise ValueError("mask_id and mask_name may not be both None")
        elif mask_id is not None and mask_name is not None:
            raise ValueError("mask_id and mask_name may not be given both")
        elif mask_name is not None:
            #only mask_name is given
            #get mask_id
            mask_id = self.names_masks.loc[self.names_masks["name"]==mask_name, "obj"].iloc[0]
            
        #delete mask
        #remark that it's impossible to remove a mask under construction
        #since a mask under construction is automatically confirmed when
        #clicking in the object_canvas
        
        #delete name of mask
        self.names_masks = self.names_masks.loc[self.names_masks["obj"] != mask_id,]
        self.names_masks = self.names_masks.reset_index(drop=True)
        
        #delete points of object
        df = self.annotation_points_masks
        df = df.loc[df['obj'] != mask_id,]
        self.annotation_points_masks = df.reset_index(drop=True)
        
        #reset active object (because it is deleted):
        self.reset_active_mask()
        
        #set saved status to False
        self.currently_saved = False
        
        self.update_image(mode=0)
        
    def delete_mask_point(self):
        """
        Delete the currently active point of mask
        """
        
        if not self.master.object_canvas.masks_visible:
            #if masks are not visible, no modifications to the masks may be done
            return
        
        if self.point_mask_active:
            #set saved state to false
            self.currently_saved = False
            
            #set current_mask_confirmed state to False
            self.current_mask_confirmed = False
            
            #update position of currently active point
            self.points_mask = np.delete(self.points_mask, self.point_id, axis=0)
            
            #update the shown image
            self.update_image(mode=1)
        
    def reset_active_mask(self):
        
        self.points_mask = np.empty(shape=(0,2))
        
        #reset mask id
        if len(self.names_masks["obj"])>0:
            self.mask_id = self.names_masks["obj"].to_numpy().max() + 1
        else:
            #there is no object yet confirmed
            self.mask_id = 0

    def button_1_release(self, event):
        """
        Actions to take when left mouse button is released

        If a new keypoint was created, the search for the next keypoint may be
        activated

        If a keypoint was activated and there is clicked in another spot of the
        image, the search for the next keypoint may be activated
        """

        #check if an image was loaded
        if self.image_name is None:
            return

        self.mouse_x = event.x
        self.mouse_y = event.y

        if self.new_keypoint_created:
            self.new_keypoint_created = False

        if not self.keypoint_reactivated:
            self.activate_next_keypoint_searching()

    def delete_object(self, obj_name=None, obj_id=None):
        """
        Delete the object of which there is currently a keypoint selected
        
        obj_id : object id, as specified in self.names
        """
        if (obj_id is None) and (obj_name is None):
        
            if not self.keypoint_reactivated:
                #There is currently no keypoint (and thus no object) selected
                tkmessagebox.showinfo("Delete object",
                                      "Select first a keypoint of the object you " +\
                                          "want to delete")
            else: #self.keypoint_reactivated:
                #There is currently a keypoint (and thus also an object) selected
                if self.annotations.shape[0] > 1:
                    #drop row of current object in dataframe
                    self.annotations = self.annotations.drop(self.object, axis=0)
                    
                    #drop row of current object in names
                    self.names = self.names.loc[self.names["obj"] != self.object,:]
                    
                else: #self.annotations.shape[0] == 1:
                    #reset dataframe
                    self.annotations = pd.DataFrame(columns=self.annotations.columns,
                                                    dtype=float)
                    new_row = pd.DataFrame(columns=self.annotations.columns,
                                           index=[0],
                                           dtype=float)
                    self.annotations = pd.concat([self.annotations, new_row],
                                                 ignore_index=True)
                    
                    #reset names
                    self.names = pd.DataFrame(data=[0, '0'],
                                              columns=["obj", "name"])
                
        else: #(obj_id is not None) or (obj_id is not None):
            
            #set active objecty correctly            
            if obj_id is None:
                self.update_active_object(obj_name=obj_name)
            elif obj_name is None:
                self.update_active_object(obj_id=obj_id)
            else:
                #both obj_name and obj_id are not None
                #the next statement will invoke an error, since this is not allowe
                self.update_active_object(obj_id=obj_id, obj_name=obj_name)
                
            #delete active object
            if self.annotations.shape[0] > 1:
                #drop row of current object in dataframe
                self.annotations = self.annotations.drop(self.object, axis=0)
                
                #drop row of current object in names
                self.names = self.names.loc[self.names["obj"] != self.object,:]
                
            else: #self.annotations.shape[0] == 1:
                #reset dataframe
                self.annotations = pd.DataFrame(columns=self.annotations.columns,
                                                dtype=float)
                new_row = pd.DataFrame(columns=self.annotations.columns,
                                       index=[0],
                                       dtype=float)
                self.annotations = pd.concat([self.annotations, new_row],
                                             ignore_index=True)
                
                #reset names
                self.names = pd.DataFrame(data=[0, '0'],
                                          columns=["obj", "name"])

        #the activated keypoint (if present) is deleted, so there is no keypoint
        #any more activated
        self.keypoint_reactivated = False
        
        #remove NaN objects
        self.remove_nan_objects()

        #reset index of self.annotations
        self.annotations.reset_index(drop=True)

        #set state of self.currently_saved to False
        self.currently_saved = False

        #initiate new object
        self.new_object()

        #update image
        self.update_image(mode=0)

    def close_image(self):
        """
        Close the current image
        """

        if self.currently_saved is False:
            #there are currently unsaved changes in the annotations
            message = "do you want to save the annotation changes you made " +\
                      "for the current image?"
            answer = tkmessagebox.askyesnocancel("Save modifications",
                                                 message=message)
            if answer is True:
                #Save changes
                self.save()

                #set state of self.currently_saved to True
                self.currently_saved = True

        #reset annotation_canvas
        self.reset_parameters()

        self.image_name = None
        self.image = None
        self.image_inter = None
        self.image_shown = None

        #update image
        self.update_image(mode=0)

    def prepare_for_skeleton_mode(self):
        """
        Prepare the AnnotationCanvas to switch to skeleton mode
        """
        #check if the current changes in self.annotations should be saved
        if not self.currently_saved and self.image_name is not None:
            #there are currently unsaved changes in the annotations
            message = "do you want to save the annotation changes you made " +\
                      "for the current image?"
            answer = tkmessagebox.askyesnocancel("Save modifications",
                                                 message=message)
            if answer is True:
                #Save changes
                self.save()

            #re-import image
            self.import_image()

    def motion(self, event):
        """
        Decisive method to invoke (or not) methods when the mouse moves

        Parameters
        ----------
        event : tkinter.event
            Motion event
        """
        if self.object_canvas_active:
            self.activate_next_keypoint_searching()
            self.object_canvas_active = False
        if not self.keypoint_reactivated:
            self.keypoint_searching(event)
            
    def rename_object(self, current_name, new_name):        
        self.names.loc[self.names["name"] == current_name, "name"] = new_name
        
    def rename_mask(self, current_name, new_name):
        self.names_masks.loc[self.names_masks["name"] == current_name, "name"] = new_name
        
    def update_active_object(self, obj_id=None, obj_name=None):
        """
        Change the currently active object
        """
        
        #process arguments
        if obj_id is None and obj_name is None:
            raise ValueError("obj_id and obj_name may not be both None")
        elif obj_id is not None and obj_name is not None:
            raise ValueError("obj_id and obj_name may not be given both")
        elif obj_name is not None:
            #only obj_name is given
            #get obj_id
            obj_id = self.names.loc[self.names["name"]==obj_name, "obj"].iloc[0]
        
        #set id of active object
        self.object = obj_id #id of current object
        
        self.update_image(mode=3)
            