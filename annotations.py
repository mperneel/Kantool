# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:59:22 2022

@author: Maarten

Definition of the Annotations class. The Annotation class contains all annotation
data and methods to modify these data.
"""
#%% packages
import pandas as pd
import numpy as np
import cv2
import os
import json
import tkinter.messagebox as tkmessagebox

#%%
class Annotations():
    
    def __init__(self, master):
        
        self.master = master
        
        self.currently_saved = True #saved status
        self.point_active = False #is there an active point?
        self.skeleton = master.skeleton
        self.annotations = None
        self.marginal_keypoint_index = -1
        self.keypoint_index = 0
        self.object = 0
        self.currently_saved = True
        self.object_memory = None
        self.new_keypoint_created = False
        self.current_mask_confirmed = True #is the current mask confirmed?
        
        self.image_name = None #image name
        self.image = None #image matrix
        
        self.keypoint_index_memory = 0
        self.keypoint_reactivated = False
        self.point_mask_reactivated = False
        
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
        
    def import_image(self, image_name):
        """
        Executive method to import image and load annotations (if available)
        """
        
        #reset parameters
        self.reset_parameters()
        self.reset_masks()

        #import image
        image = cv2.imread(image_name)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_name = image_name

        #import annotations (if they exist)
        if image_name.split('.')[0] + '.csv' in os.listdir():
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
                
                #when the application was previously closed incorrectly, some
                #annotation files can get corrupted (too many or too few columns)
                #the if statement belows handles those corrupted data
                if self.annotations.shape[1] > len(columns):
                    self.annotations = self.annotations.iloc[:, :len(columns)]
                elif self.annotations.shape[1] < len(columns):
                    n = len(columns) - self.annotations.shape[1] 
                    self.annotations.loc[:,columns[-n:]] = np.nan                    
                    
                self.annotations.columns = columns
            except pd.errors.EmptyDataError:
                #create empty dataframe
                self.annotations = pd.DataFrame(columns=columns,
                                                dtype=float)
                empty_row = pd.DataFrame(columns=self.annotations.columns,
                                         index=[0],
                                         dtype=float)
                self.annotations = pd.concat([self.annotations, empty_row],
                                             ignore_index=True)                
              
            #create generic names for objects
            names_array = np.arange(len(self.annotations.index))
            names_dict = {"obj": names_array.copy(),
                          "name": [str(i) for i in names_array]}
            self.names = pd.DataFrame(names_dict)
        
        #import masks (if they exist)
        self.load_masks()
        
        return       
        
    def reset_parameters(self):
        """
        Reset attributes to their default value
        """
        
        self.image = None
        self.image_name = None
        
        self.object = 0
        self.keypoint_index = 0
        self.marginal_keypoint_index = -1

        #reset annotations
        if self.skeleton.n_keypoints > 0:
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
            
    def new_object(self):
        """
        Create a new object
        """
        if self.skeleton.n_keypoints is None :
            #if self.skeleton.n_keypoints is None, this means there is currently
            #no project loaded
            
            #therefore, we return immediately
            return
        
        self.marginal_keypoint_index = 0
        self.keypoint_index = self.skeleton.func_annotation_order[self.marginal_keypoint_index]
        
        #remove all objects with only NaN coordinates
        self.remove_nan_objects()
    
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
        
    def new_mask(self):
        """
        Confirm current mask
        """
                    
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
                
    def new_keypoint(self, x, y):
        """
        Create a new keypoint

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        df = self.annotations        

        df.iloc[self.object, self.keypoint_index * 2] = x
        df.iloc[self.object, self.keypoint_index * 2 + 1] = y

        self.annotations = df
        self.new_keypoint_created = True

        #set state of self.currently_saved to False
        self.currently_saved = False
    
    def activate_keypoint(self, object_id, keypoint_id):       
        
        #re-activate a keypoint
        self.keypoint_reactivated = True

        #assign memory attributes
        self.keypoint_index_memory = self.keypoint_index
        self.object_memory = self.object
        
        #assign active index attributes
        self.keypoint_index = keypoint_id
        self.marginal_keypoint_index = \
            np.where(self.keypoint_index == \
                     np.array(self.skeleton.func_annotation_order))[0][0]
        #np.where() returns a tuple containing a numpy array
        #with the indices for which the condition is True, we
        #need the first element of that matrix (which is the
        #first and only element of the tuple)
        self.object = object_id
        
        return
        
    def keypoint_distance(self, loc):
        """
        Calculate distances between given keypoints and all annotated keypoints

        Parameters
        ----------
        loc : TYPE
            location (x,y) coordinates

        Returns
        -------
        None.

        """
        
        #get number of keypoints
        n_keypoints = self.skeleton.n_keypoints
        
        #get keypoint coordinates
        coordinates = self.annotations.to_numpy()
        
        #pre-allocate distance matrix
        distance = np.zeros(shape=(len(self.annotations), n_keypoints))
        
        #calculate all distances
        for i in range(n_keypoints):
            coordinates_i = coordinates[:, 2*i:2*i+2]
            distance_i = np.sqrt(np.sum((coordinates_i - loc)**2, axis=1))
            distance[:,i] = distance_i
        
        return distance
            
    def closest_keypoint(self, loc):
        """
        

        Parameters
        ----------
        loc : TYPE
            Location (x,y) coordinates

        Returns
        -------
        object_id : TYPE
            DESCRIPTION.
        keypoint_id : TYPE
            DESCRIPTION.
        distance : TYPE
            DESCRIPTION.

        """
        
        distances = self.keypoint_distance(loc)
        
        if (np.sum(np.isnan(distances)) < np.prod(distances.shape)):
            #look for smallest real number in array
            min_index = np.nanargmin(distances.flatten())
            
            #unravel object and keypoint id
            object_id, keypoint_id = np.unravel_index(min_index,
                                                      shape=distances.shape)
            
            #look up the corresponding distance
            distance = distances[object_id, keypoint_id]        
            
        else:
            #no keypoints were yet drawn
            object_id, keypoint_id, distance = None, None, None
        
        return (object_id, keypoint_id, distance)
    
    def activate_next_missing_keypoint(self):
        """
        Update all attributes so the next missing keypoint can be looked for.
        If an object is complete, a new object will be created
        """
        
        #if currently a keypoint was re-activated, end it's re-activation state
        if self.keypoint_reactivated:
            self.keypoint_reactivated = False
            
        #if current object is not last object and is empty, delete nan-objects
        #and create a new object
        if self.current_object_empty and\
            self.object != self.n_objects - 1:
            self.remove_nan_objects()
            self.new_object()
            return            
        
        #No keypoint was re-activated currently, look for the next missing keypoint
        #if no keypoint is missing, create a new object
        missing_keypoint_found = False
        n_keypoints = self.skeleton.n_keypoints
        loop = 0

        while not missing_keypoint_found:
            loop += 1

            self.marginal_keypoint_index = (self.marginal_keypoint_index + 1) %\
                n_keypoints
            self.keypoint_index = self.skeleton.keypoint_index(self.marginal_keypoint_index)
            if np.isnan(self.annotations.iloc[self.object, 2*self.keypoint_index]):
                missing_keypoint_found = True
            elif loop == n_keypoints:
                #confirm current object and create new object
                self.new_object()
                return
        
    def delete_mask(self, mask_id=None, mask_name=None):
        
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
        return
        
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
            if self.image_name.split(".")[0] + "_mask.json" in os.listdir(self.master.wdir):
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
            
    def new_mask_point(self, x, y, sensitivity=10, magnetic_border=None):
        """
        add a new point to the mask  
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        sensitivity : TYPE
            sensitivity, expressed as number of pixels on the image 
        magnetic_border : Int
            size of the magnetic border, expressed as number of pixels on the
            image

        Returns
        -------
        bool
            DESCRIPTION.

        """
        
        #preprocess input arguments
        s = sensitivity
        m = magnetic_border
        
        #set saved state to false
        self.currently_saved = False
        
        #set object_confirmed state to False
        self.current_mask_confirmed = False        
        
        #set point_active to False
        self.point_active = False
        
        #Check if a point in the image is activated and not a point
        #in the surrounding grey area                   
        
        self.point_mask_active = True
        
        #boolean to indicate if an extra point should be added to the current mask
        add_extra_point = False
        
        if len(self.points_mask)>0:
            #check if you have to replace a yet existing point or if you have
            #to create a new point
            dist_to_points = np.sqrt(np.sum(np.square(\
                        self.points_mask - np.array([[x, y]])), axis=1).\
                                     astype(float))
            
            if min(dist_to_points) <= s:
                #replace a yet existing point
                self.point_id = np.argmin(dist_to_points)
            else:
                #add an extra point
                if len(self.points_mask) >=3:
                    #check if you have to add an extra point at the end, or you have to
                    #split an edge

                    #calculate orthogonal distance to edges
                    C = np.array([[x, y]])
                    points = self.points_mask.astype(float)

                    A = points
                    B = np.concatenate((points[1:], [points[0]]), axis=0)
                    u = B-A
                    v = np.concatenate((-u[:,1].reshape(-1,1), u[:,0].\
                                        reshape(-1,1)), axis=1)                    
                    v = v / np.linalg.norm(v, axis=1).reshape(-1,1)

                    ac = A - C #vectors from A to C

                    #orthogonal distances to edges
                    d = np.abs((ac * v).sum(-1))

                    if min(d)<= s:
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
                            add_extra_point = True
                    else:#min(d)>dist_max
                        #add extra point at the end of self.points
                        add_extra_point = True
                else: #len(self.points)<3:
                    #add extra point at the end of self.points
                    add_extra_point = True
        else: #len(self.points)==0:
            #add first point in self.points
            add_extra_point = True
        
        #it makes no sense to define a separate function for the code below,
        #which adds an extra point at the end of the mask 
        #therefore, we execute point addition in a if statement at the end of
        #this function, in which the condition is a boolean defined in the if-else
        #structure of the previous part of the function code
        if add_extra_point:
            #if a point is close to the borders, adapt x and y, so point is
            #drawn on the borders
            w = self.image.shape[1]
            h = self.image.shape[0]
            if x <= m:
                x = 0
            elif w - x <= m:
                x = w - 1
                
            if y <= m:
                y = 0
            elif h - y <= m:
                y = h - 1
            
            #add point to mask
            self.point_id = len(self.points_mask)
            self.points_mask = np.append(self.points_mask,[[x, y]], axis=0)

        return

    def get_mask_id(self, name):
        """
        Get the id (identifier index) of a mask

        Parameters
        ----------
        name : str
            name of the mask

        Returns
        -------
        idx : int
            identifier index of the mask
        """
        
        idx = self.names_masks.loc[self.names_masks["name"]==name, "obj"].iloc[0]
        return idx
    
    def get_object_id(self, name):
        """
        get the id (identifier index) of an object

        Parameters
        ----------
        name : str
            name of the object
        """
        
        idx = self.names.loc[self.names["name"]==name, "obj"].iloc[0]
        return idx
    
    def rename_object(self, current_name, new_name):        
        self.names.loc[self.names["name"] == current_name, "name"] = new_name
        
    def rename_mask(self, current_name, new_name):
        self.names_masks.loc[self.names_masks["name"] == current_name, "name"] = new_name
        
    def update_keypoint(self, x, y):
        """
        Adapt the coordinates of the current keypoint

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate      
        """
        
        i = self.keypoint_index

        #update keypoint coordinates
        self.annotations.iloc[self.object, i * 2] = x
        self.annotations.iloc[self.object, i * 2 + 1] = y

        #set state of currently_saved to False
        self.currently_saved = False
        
    def delete_keypoint(self):
        """
        Delete the currently activated keypoint
        """
        
        if self.keypoint_reactivated:
            i = self.keypoint_index
            self.annotations.iloc[self.object, i * 2: i * 2 + 2] = np.nan
            self.keypoint_reactivated = False
            #decrease self.marginal_keypoint_index with 1 so
            #self.activate_next_keypoint_searching will 'activate' the search
            #to a keypoint of the type of the deleted keypoint
            self.marginal_keypoint_index -= 1

            #set state of self.currently_saved to False
            self.currently_saved = False
            
    def delete_mask_point(self):
        """
        Delete current point of mask
        """
        if self.point_mask_active:
            #set saved state to False
            self.currently_saved = False
            
            #set current_mask_confirmed state to False
            self.current_mask_confirmed = False
            
            #update position of currently active point
            self.points_mask = np.delete(self.points_mask, self.point_id, axis=0)
            
    def update_active_mask(self, mask_id):
        """
        Update the attributes of the active mask to those corresponding to the 
        mask with identifier index mask_id

        Parameters
        ----------
        mask_id : int
            Identifier index of the new active mask
        """
        
        #set id of active mask
        self.mask_id = mask_id #id of newly active mask
        
        #set points of active mask
        self.points_mask = self.annotation_points_masks.loc[self.annotation_points_masks["obj"]==self.mask_id,
                                                            ["x", "y"]].to_numpy()
        
        #set id of last active point of mask (set to last point of mask)
        self.point_id = len(self.points_mask) - 1
        
    def update_point_mask(self, x, y):
        """
        Update the coordinates of the currently active mask point

        Parameters
        ----------
        x : float
            x coordinate (horizontal axis, left origin)
        y : float
            y coordinate (vertical axis, top origin)
        """
        #update position of currently active point
        self.points_mask[self.point_id,:] = np.array([x, y])
            
    def reset_active_mask(self):
        """
        Reset attributes of active mask
        """
        #reset points_mask
        self.points_mask = np.empty(shape=(0,2))
        
        #reset mask id
        if len(self.names_masks["obj"])>0:
            self.mask_id = self.names_masks["obj"].to_numpy().max() + 1
        else:
            #there is no mask yet confirmed
            self.mask_id = 0
            
    def delete_object(self, obj_name=None, obj_id=None):
        """
        Delete the currently active object
        
        Only one of the inputs obj_name and obj_id may be specified

        Parameters
        ----------
        obj_name : str, optional
            Object name. The default is None.
        obj_id : int, optional
            object identifier integer. The default is None.
        """
        
        #activate correct object
        if (obj_id is None) and (obj_name is None):
        
            if not self.keypoint_reactivated:
                #There is currently no keypoint (and thus no object) selected
                tkmessagebox.showinfo("Delete object",
                                      "Select first (a keypoint of) the object you " +\
                                          "want to delete")
            #else: #self.keypoint_reactivated:
                #There is currently a keypoint (and thus also an object) selected               
                
        else: #(obj_id is not None) or (obj_id is not None):
            
            #set active objecty correctly
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
            obj_id = self.get_object_id(obj_name)
        
        #set id of active object
        self.object = obj_id #id of current object
        
    @property
    def last_object_empty(self):
        """    
        Check if the last object is a preallocated object
        
        If the last object is a preallocated object, True is returned. If the
        last object contains at least one specified keypoint, False is returned
        """
        
        if self.annotations is None:
            return True
        
        if np.prod(np.isnan(self.annotations.iloc[-1,:].to_numpy())).astype(bool):
            #boolean product is equal to and operator
            #last object is a preallocated object
            return True
        #last object is no preallocated object, but contains at least one 
        #specified keypoint
        return False
        
    @property
    def current_object_empty(self):
        """    
        Check if the current object is a preallocated object
        
        If the current object is a preallocated object, True is returned. If the
        last object contains at least one specified keypoint, False is returned
        """
        
        if self.annotations is None:
            return True
        
        if np.prod(np.isnan(self.annotations.iloc[self.object,:].to_numpy())).astype(bool):
            #boolean product is equal to and operator
            #last object is a preallocated object
            return True
        #last object is no preallocated object, but contains at least one 
        #specified keypoint
        return False
    
    @property
    def n_objects(self):
        """
        Number of defined objects
        """
        if self.annotations is None:
            return 0
        return len(self.annotations)
            
        
        