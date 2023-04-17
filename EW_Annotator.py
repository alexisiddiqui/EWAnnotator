import os
import imageio as iio
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import subprocess
import librosa.display
import soundfile as sf




#probs only used for rendering video
import cv2

#load json of shows

### TODO List ###
# Parser
# write docstrings
# proper time conversion





## TODO Fxi this for the unittests
def json_to_pandas(json_path, jsontype=None):
    """Loads a json file into a pandas dataframe"""
    if jsontype == None:
        jsontype= "shows"
    with open(json_path, "r") as f:
        dict = json.load(f)

    df = pd.DataFrame(dict[jsontype])
    return df

def pandas_to_json(df, json_path, jsontype="shows"):
    """Saves a pandas dataframe to a json file"""
    dict = {jsontype: df.to_dict(orient="records")}

    with open(json_path, "w") as f:
        json.dump(dict, f)


#add shows from df to new dictionary as images
def image_to_dict(df, folder="RAW"):
    """Loads images from a folder into a dictionary"""
    image_dict = {}
    for i in range(len(df)-1):
        df_row = df.iloc[i]
        image_path = os.path.join(folder, df_row["path"] )
        image_dict[df_row["showName"]] = iio.imread(image_path)
    return image_dict

#save images from dictionary as pngs without json

### TODO NAMES!!!! or show names AGHHHHH this has become complicated fuck
def save_images_from_dict(show_image_dict, df=None, folder="custom/shows"):
    """Saves images from dictionary using file names from dataframe if present"""
    for show_name, image in show_image_dict.items():
        if df is not None:
            row =  df.loc[df['showName'] == show_name]
            show_name = row["path"].values[0]
        else:
            show_name = show_name + ".png"

        image_path = os.path.join(folder, show_name)
        iio.imwrite(image_path, image)

# get frame rate easily
def df_get_frameRate(df, name):
    """Returns framerate from a named show in a given dataframe"""
    df = df[df["showName"] == name]
    return df["frameRate"][0]


def convert_show_framerate(show_image, dict=None, old_frameRate=None, new_frameRate=64):
    """Convert framerate of show iamge from a given dictionary to new framerate, defualting to 64"""
    if old_frameRate == None:
        raise ValueError("Set old_frameRate")

    try:
        show_image = dict[show_image]
    except:
        pass

    scale = new_frameRate/old_frameRate
    old_image_shape = show_image.shape
    new_image= np.resize(show_image, (int(old_image_shape[0]*scale), old_image_shape[1]))

    return new_image


### TODO ### Time conversion - TEST if it works - add beats to frames
def show_seconds_to_frames(seconds, frameRate=64, BPM=None):
    """Converts seconds to frames for a given framerate and BPM"""
    if BPM == None:
        return "Set BPM"
    framesperbeat = frameRate/4
    BPS = BPM/60
    framespersecond = (framesperbeat*BPS)

    frames = np.copy(seconds)
    nframes = frames*framespersecond

    return nframes.astype(int)

def beats_to_seconds(beats, BPM=None):
    """Converts beats to seconds for a given BPM"""
    if BPM == None:
        return "Set BPM"
    secondsperbeat = 60/BPM
    seconds = np.copy(beats)
    seconds = seconds*secondsperbeat

    return seconds.astype(float)

def show_beats_to_frames(beats, frameRate=64, BPM=None):
    """Converts number of beats to number of frames for a given framerate and BPM"""
    if BPM == None:
        return "Set BPM"
    seconds = beats_to_seconds(beats, BPM)
    frames = show_seconds_to_frames(seconds, frameRate, BPM)

    return frames.astype(int)


#visualise audio sequence

#covnerts mp3 to wav and loads it
def convert_mp3_to_wav(mp3path, wavpath=None):
    """Converts mp3 to wav and loads it, uses same name if no wavpath is given"""
    if wavpath == None:
        wavpath = mp3path.replace(".mp3", ".wav")
    subprocess.call(['ffmpeg', '-y', '-i', mp3path , wavpath])
    
    return librosa.load(wavpath)

def load_wav(path, start=None, stop=None):
    """Loads a wav file from a given path"""
    ## TODO: add time conversions
    y, sr = librosa.load(path)
    return y[start:stop], sr

#visualise waveform
def visualise_audio(audio, start=None, stop=None):
    ## TODO: add time conversions
    try:
        y, sr = librosa.load(audio)
    except:
        y, sr = audio
    y=y[start:stop]

    plt.figure(figsize=(7, 3))
    plt.title('waveform')
    librosa.display.waveshow(y, sr=sr)

def visualise_spectrogram(audio, start=None, stop=None):
    """Visualises a spectrogram of a given audio file"""
    ## TODO: add start and stop regions
    ### TODO add config file for parameters
    try:
        y, sr = librosa.load(audio)
    except:
        y, sr = audio
    y=y[start:stop]
    plt.figure(figsize=(7, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

### Import and visualise lighting sequences

def load_lighting_sequence(path, folder="RAW"):
    """Loads a lighting sequence from a given path and root folder"""
    ### TODO make this more general - add an else statement for folder is none in case no folder is given
    if folder is not None:
        path = os.path.join(folder, path)
    img = iio.imread(path)
    return img

#visualise lighting sequence

def visualise_lighting_sequence_object(img, dict=None):
    """Plots a lighting sequence image object"""
    # works either with an image object or image from dictionary
    ### TODO add more customisations
    plt.figure(figsize=(7,3))

    try:
        img = dict[img]
    except:
        pass
    plt.imshow(img)




# Concatenate show units

#create show sequence segments
def show_segment(show, start=None, stop=None):
    """Takes a show image and returns a segment of it"""
    # TODO - add time conversions
    show = show[start:stop,:,:]
    return show

#concatenate show segments
def concatenate_show_units(show_list):
    """Takes a list of show images and concatenates them to produce a show sequence"""
    # We can make this more flexible later by not having the user create the segments first
    # takes list of show image nparray and concatenates them to produce show sequence
    show_sequence = np.concatenate(show_list, axis=1)
    return show_sequence

### TODO - TEST
def visualise_lights(images, positions, resoltuion):
    #takes list of images and positions and draws them on a canvas
    if len(images) != len(positions):
        raise ValueError("Number of images and positions must be equal lists")
    canvas = np.zeros((resoltuion[0],resoltuion[1],4))

    for idx, image in enumerate(images):
        canvas[positions[idx][0]:positions[idx][0]+image.shape[0], 
               positions[idx][1]:positions[idx][1]+image.shape[1]] = image
   
    return canvas

def tune_lights(image, weights=None):
    #takes list of images and weights and returns weighted images
    if weights == None:
        weights = np.ones_like(image)
    if weights.shape != image.shape:
        raise ValueError("Image and weights must be same shape")

    image = image*weights

    return image




### class for show segments - to be used in show sequence
# returns dictionary of shows
class Show_library:
    ### Holds the library of shows
    # information is passed through json file and held in a pandas dataframe
    # images are held in a dictionary
    def __init__(self, json_path=None,folder=None) -> None:
        self.show_image_dict = {}
        self.df = pd.DataFrame(columns=["showName", "path", "numLights", "frameRate"])
        self.show_names = []
        if json_path != None:
            self.load_json(json_path, folder)
        pass
    
    def __call__(self):
        return self.show_image_dict
    
    def __str__(self) -> str:
        try:
            #check this actually prints list
            print(self.show_names)
        except:
            return "No shows loaded"

    def load_json(self, json_path, folder=None):
        self.df = pd.merge(self.df, json_to_pandas(json_path), how="outer")
        self.show_image_dict.update(image_to_dict(self.df, folder))
        self.show_names = list(self.show_image_dict.keys())

    ### TODO BROKEN add names/indexes 
    def save_library(self, json_path, output_folder="custom"):
        pandas_to_json(self.df, json_path)
        save_images_from_dict(self.show_image_dict, 
                              df=self.df, 
                              folder=output_folder)

    def add_show(self, show_name, image, path=None, numLights=16, frameRate=64):
        
        self.show_image_dict[show_name] = image
        self.show_names = list(self.show_image_dict.keys())


        if path != None:
            if path.endswith(".png")==False:
                path = path + ".png"
        
        new_df = pd.DataFrame({"showName": show_name, "path": path, "numLights": numLights, "frameRate": frameRate }, index=[0])

        self.df = pd.merge(self.df, 
                            new_df,
                            how="outer")

    # TODO make this work with multiple show names
    def remove_show(self, show_name):      
        self.show_image_dict.pop(show_name)
        self.show_names.remove(show_name)
        self.df = self.df[self.df["showName"] != show_name]

    # TODO make this work with multiple show names?
    def get_show(self, show_name):
        return self.show_image_dict[show_name]
    
    # TODO make this work with multiple show names?
    def get_show_info(self, show_name):
        return self.df[self.df["showName"] == show_name]


    def set_framerate(self, show_name=None, new_frameRate=64):
        #converts images to new frame rate
        #set df to new frame rate
        #either sets it for a single show or applies to all the shows
        if show_name == None:
            for name in self.show_names:
                convert_show_framerate(name, self.show_image_dict, new_frameRate)
                show = self.df[self.df["showName"] == name]
                show["frameRate"] = new_frameRate
                self.df[self.df["showName"] == name] = show

        else:
            convert_show_framerate(show_name, self.show_image_dict, new_frameRate)
            show = self.df[self.df["showName"] == show_name]
            show["frameRate"] = new_frameRate
            self.df[self.df["showName"] == show_name] = show



### class for lighting system
#holds library of lights
# TODO give default positions

class Light_show_setup:
    def __init__(self, numLights=None, json_path=None, folder=None) -> None:
        if numLights==None:
            self.numLights = 16
        self.light_dict = {}
        self.df = pd.DataFrame(columns=["name", "path", "lightnumber"])
        self.light_setup = pd.DataFrame(columns=["lightnumber", "image", "position"])
        self.stage_size = (640, 480)
        if json_path != None:
            self.load_json(json_path, folder)
        pass

    def __call__(self):
        try: ### CHange this so something that can be used elegantly inside a function
            return self.render_scene(self)
        except:
            return "No light setup loaded"
        
    def __str__(self) -> str:
        try:
            return self.light_setup["lightnumber"][0]
        except:
            return "No light setup loaded"
        

    def load_json(self, json_path, folder=None):
        self.df = pd.merge(self.df, 
                           json_to_pandas(json_path, jsontype="lights"), 
                           how="outer")
        self.light_dict.update(image_to_dict(self.df, 
                                             folder=folder))


    def add_light(self, name, image=None, path=None, lightnumber=None):
        # must have a name
        if image and path == None:
            raise ValueError("Must specify image and/or path")

        if image == None:
            image = iio.imread(path)
        elif path == None:
            path = "lights" 
            path = os.path.join(path, name + ".png")

        self.light_dict[name] = image
        new_light = pd.DataFrame([{"name":name, "path":path, "lightnumber":lightnumber}])
        self.df = pd.merge(self.df, new_light, how="outer")
    
    def remove_light(self, name):
        self.light_dict.pop(name)
        self.df = self.df[self.df["name"] != name]

    def save_library(self, json_path, output_folder="custom"):
        # TODO check that this actually works - not sure if this works for 
        pandas_to_json(self.df, json_path, jsontype="lights")
        output_folder = os.path.join(output_folder, "lights")

        save_images_from_dict(self.light_dict, 
                              df=self.df, 
                              folder=output_folder)



        # pandas_to_json(self.df, json_path, "lights")
        # output_folder = os.path.join(output_folder, "lights")
        # save_images_from_dict(self.light_dict, output_folder)

    def set_light(self, name, light_number=None):
        image = self.light_dict[name]
        light_set = pd.DataFrame([{"name":name,"image": image, "lightnumber": light_number}])
        pd.concat([self.light_setup, light_set])
        self.numLights = len(self.light_setup["lightnumber"])   


    def set_positions(self, light_numbers=None, positions=None):
        if light_numbers == None:
            #defaults to 16
            light_numbers=range(self.numLights)

        self.light_setup["lightnumber"] = {"light_number": light_numbers, "position": positions}
    
    ### TODO do parse sequence - find out how show format works
    def parse_show_sequence(self, show_sequence):
        #takes show sequence and weights

        #split seqence into channels for each of the lights

        #parse show_sequence based on red channel- divide value by 255
        weights = show_sequence[:,:,0]/255

        try:
            return weights
        except:
            raise Exception("No show sequence loaded or problem with show sequence")

    def render_scene(self, weights=None):
        images = self.light_setup["image"]
        # does this return a list of images or a np array?

        positions = self.light_setup["position"]
        ### TODO ### Does this actually work with the weights
        ### -> do we want to allow weights that arent the same
        if weights != None:
            for idx, image in enumerate(images): 
                images[idx] = tune_lights(image, weights[idx])

        canvas = visualise_lights(images, positions, self.stage_size)

        return canvas



### class for audio and final show sequences
# must take in a song and returns show sequence
class Show_sequence:
    def __init__(self, path=None) -> None:
        self.show_dict = {}
        self.df = pd.DataFrame()
        self.show_sequence = {}
        self.frameRate = 64
        if path != None:
            self.load_song(path)
        pass

    def __call__(self):
        return self.show_sequence

    def __str__(self) -> str:
        try:
            return self.song_name
        except:
            return "No song loaded"

    def load_song(self, path, start=None, stop=None):
        ### TODO ### -> add in start and stop time conversions!!!

        if path.split(".")[-1] == "mp3":
            path = convert_mp3_to_wav(path)
        path.replace(".mp3", ".wav")
        self.audio, self.sr = load_wav(path, start, stop)
        self.song_name = path.split("/")[-1].replace(".wav", "")
        self.length = (self.audio.shape[0])/self.sr
        self.bpm, self.beat_times = librosa.beat.beat_track(self.audio, sr=self.sr)
        self.beat_times = np.concatenate(([0], self.beat_times))
    ### TODO ### -> add in start and stop time conversions!!!

    def insert_show(self, show_image, show_library=None, start=None, stop=None, index=None):
        try:
            show_image = show_library[show_image]
        except:
            pass
        show_image = show_image[start:stop]
        ### could try  changing show_sequence to a dataframe? can then search based on index
        # although unsure if this will be faster
        if index == None:
            self.show_sequence = np.append(self.show_sequence, show_image, axis=0)
        else:
            self.show_sequence = np.insert(self.show_sequence, index, show_image, axis=0)

    def remove_show(self, index):
        self.show_sequence = np.delete(self.show_sequence, index, axis=0)

    def save_sequence(self, path):
        iio.imwrite(path, self.show_sequence)

    def visualise_spectrogram(self, start=None, stop=None):
        visualise_spectrogram(self.audio[start:stop])
    
    def visualise_waveform(self, start=None, stop=None):
        visualise_audio(self.audio[start:stop])

    def visualise_sequence(self, start=None, stop=None):
            visualise_lighting_sequence_object(self.show_sequence)           


    
    ### TODO ### -> complete show sequence to show both audio and lighting on the same axes
    ### TIME CONVERSION!!!
    def visualise(self, start=None, stop=None):
        pass

    ### TODO ### -> finish render video
    # Make lighting setup
    def render_show(self, output_path, Light_show_setup, start=None, stop=None, fps=30):
        if self.audio == None:
            raise Exception("No audio loaded")

        #calculate time
        length_in_seconds = self.audio.shape[0]/self.sr
        #fill in array to match length of audio
        blank_frames = show_seconds_to_frames(length_in_seconds, self.bpm)
        blank = np.zeros((blank_frames,self.audio.shape[1:]))
        
        #concatenate show with blank
        output_show = np.concatenate((self.show_sequence, blank), axis=0)
        output_light_frames = np.array([show_seconds_to_frames(time, self.bpm) for time in self.beat_times])

        #parse show
        try:
            weights = Light_show_setup.parse_show(self.show_sequence)
        except:
            raise Exception("Light show setup not loaded properly")
        
        #render frames
        rendered_frames = np.empty(output_show.shape[0])
        for idx,image in enumerate(output_show):
            rendered_frames[idx] = Light_show_setup.render_frame(image, weights[idx])

        #average frames betwen average_light_frames to account for frame rate
        averaged_frames = np.empty(output_light_frames.shape[0])

        for idx, frame in enumerate(output_light_frames):
            averaged_frames[idx] = np.mean(rendered_frames[frame:output_light_frames[idx+1]], axis=0)

        self.averaged_frames = averaged_frames

        frames = np.uint8(self.averaged_frames*255)

        height, width, _ = frames.shape
        
        #make video
        audio_path = "temp_audio.wav"
        print("Writing temp audio to file: "+audio_path)
        sf.write(audio_path, self.audio, self.sr)


        command = ['ffmpeg',
                   '-y',  # overwrite output file if it exists
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f"{width}x{height}",
                   '-pix_fmt', 'rgb24',
                   '-r', f"{fps:.2f}",
                   '-i', '-',
                   '-i', audio_path,
                   '-acodec', 'aac',
                   '-vcodec', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-crf', '0',
                   output_path]

        # Start ffmpeg process and pass in frames and audio data
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        for idx, frame in enumerate(self.averaged_frames):
            # Convert frame to bytes and pass to ffmpeg process
            process.stdin.write(frame.tobytes())
            print("Writing frame: "+str(idx)+" of "+str(len(self.averaged_frames)), end="\r")


        process.stdin.close()
        process.wait()

        # Delete temporary audio file
        os.remove(audio_path)

        return output_path


### Create animated music video of sequence

###





