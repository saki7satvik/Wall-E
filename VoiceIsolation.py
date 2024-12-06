from spleeter.separator import Separator
import os


def seperate_vocals(input_audio, output_directory):
    
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_audio, output_directory)

    print(f"Separated audio saved in: {output_directory}")


def Read_Audio_seperate(input_folder):
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            seperate_vocals(input_folder + filename, 'Isolated Voice')
        else:
            continue
    print("All audio files seperated")



if __name__ == "__main__":
    Read_Audio_seperate('Raw Voice/')
