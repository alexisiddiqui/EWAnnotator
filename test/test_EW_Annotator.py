import unittest
import os

from EW_Annotator import *


shows_json_path = "shows/shows.json"

test_dir = "test" ## change this to blank as we are already in the test dir

positions = [(100,100)]




class TestEW_Annotator(unittest.TestCase):
    def test_EW(self):
        self.assertEqual(1, 1)

    def test_load_and_save_json(self):
        # Load a json file
        json_path = os.path.join(test_dir, shows_json_path)

        df = json_to_pandas(json_path)
        
        # Save it to a new file
        pandas_to_json(df, json_path)

        # Load the new file
        df2= json_to_pandas(json_path)

        # Compare the two
        self.assertTrue(df.equals(df2))



    def test_load_and_save_dict(self):
        # Load a df
        json_path = os.path.join(test_dir, shows_json_path)

        df = json_to_pandas(json_path)
        # load dict
        dict = image_to_dict(df, folder=test_dir)
        # Save it to a new file
        save_images_from_dict(dict, folder=test_dir)
        # Load the new file
        dict2 = image_to_dict(df, folder=test_dir)

        # Compare the two
        self.assertEqual(len(dict), len(dict2))
        


    def test_time_seconds_to_frames(self):
        # Test that the time conversion functions work

        #test seconds to frames for a light show

        frames = show_seconds_to_frames(1, 64, 120)

        self.assertEqual(frames, 32)

    def test_time_beats_to_seconds(self):
        
        #test beats to seconds for a light show

        seconds = beats_to_seconds(1, 120)

        self.assertEqual(seconds, 1*60/120)

    def test_time_beats_to_frames(self):

        #test beats to frames for a light show

        frames = show_beats_to_frames(1, 64, 120)

        self.assertEqual(frames, 64/2/2)



    def test_audio_loaders(self):
        # Test that the audio loading functions work

        #test loading a wav file
        wavpath = "test.wav"

        wavpath = os.path.join(test_dir, wavpath)

        wav = load_wav(wavpath)

        #test loading a mp3 file
        mp3path = "test.mp3"

        mp3path = os.path.join(test_dir, mp3path)

        mp3 = convert_mp3_to_wav(mp3path)

        self.assertEqual(wav[0].all(), mp3[0].all())

        #test loading part of a wav file
       
        wav = load_wav(wavpath, start=0, stop=1)

        self.assertEqual(wav[1], 1*wav[1])



    def test_image_loaders(self):
        # test load lighting sequence

        path = os.path.join(test_dir, "test.png")

        img = load_lighting_sequence(path, folder=None)

        self.assertEqual(img.shape, (120, 160, 4))
        ### TODO ### Visualise tests?
        # visualise lighting sequence
        visualise_lighting_sequence_object(img)

        # test show_segment
        img = show_segment(img, start=0, stop=100)

        self.assertEqual(img.shape, (100, 160, 4))

        # concatenate lighting sequences
        

    def test_visualise_stage(self):
        # test visualise lights

        path = os.path.join(test_dir, "test.png")

        img = load_lighting_sequence(path, folder=None)

        canvas = visualise_lights([img], [(100,100)], (640,480))

        self.assertEqual(canvas.shape, (640, 480, 4))

        # test tune lights

        new_img = tune_lights(img)

        self.assertEqual(new_img.shape, (120, 160, 4))

        # test visualise tuned lights
        visualise_lighting_sequence_object(new_img)


    def test_Show_library(self):
        # test load json

        json_path = os.path.join(test_dir, shows_json_path)

        show_library = Show_library(json_path, folder=test_dir)

        # test add show
        
        path = os.path.join(test_dir, "test.png")

        img = load_lighting_sequence(path, folder=None)


        show_library.add_show("test", img)

        # test get show

        self.assertEqual(img.all(), show_library.get_show("test").all())
        # test get show


        ## TODO LATER '
        # self.assertEqual(,Show_library.get_show_info("test"))


        # test remove show
        show_library.remove_show("test")
        print(show_library.show_image_dict.keys())


        # test save library


        show_library.save_library(json_path, output_folder=test_dir)

        # test set framerate


        ### TODO CHECK THIS IS WORKING
        show_library.set_framerate(new_frameRate=64)

        ### TODO how to check framerate of all the shows are the same
        

    def test_light_show(self):
        # test load json 

        json_path = shows_json_path.replace("shows", "lights")

        # json_path = os.path.join(test_dir, lights_json_path)
        
        
        light_show_library = Light_show_setup(json_path, folder=test_dir)


        # test add light

        path = os.path.join(test_dir, "test.png")

        light_show_library.add_light("test", path=path)
        
        # test set lights
        light_show_library.set_light("test", [1])
        
        print(light_show_library)


        # test set positions

        light_show_library.set_positions(1, positions)

        # test remove light

        light_show_library.remove_light("test")

        # test save library

        # light_show_library.save_library(json_path, output_folder=test_dir)




    def test_Show_sequence(self):
        # test load song

        path = os.path.join(test_dir, "Supersonic(VIP).wav")

        show_sequence = Show_sequence(path)

        self.assertEqual(show_sequence.song_name, "Supersonic(VIP)")

        json_path = os.path.join(test_dir, shows_json_path)

        # test insert show

        show_library = Show_library(json_path, folder=test_dir)



        show_sequence.insert_show("EWA1",show_library.show_image_dict)

        show_sequence.insert_show("EWA2",show_library.show_image_dict, index=0)

        # test remove show  

        show_sequence.remove_show(1)

        self.assertEqual(show_sequence(), show_library.get_show("EWA2"))

        # test save sequence

        show_sequence.save_sequence("test_sequence.png")

        # test visualisers

        show_sequence.visualise_waveform()
        show_sequence.visualise_spectrogram()
        show_sequence.visualise_sequence()

        ### TODO make visualiser work for both audio and show sequence
        show_sequence.visualise()
        pass



    def test_renderer(self):

        #load show library

        #load light library

        #load show sequence

        #render
        
        # test video

        # test audio
        
        pass
