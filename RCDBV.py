# -*- coding: utf-8 -*
import time
import serial
import pygame

# written by Ibrahim for Public use

# Checked with TFmini plus

# ser = serial.Serial("/dev/ttyUSB1", 115200)

ser = serial.Serial("/dev/ttyAMA0", 115200)
# ser = serial.Serial("COM12", 115200)


# we define a new function that will get the data from LiDAR and publish it
def read_data():
    while True:
        counter = ser.in_waiting # count the number of bytes of the serial port
        if counter > 8:
            bytes_serial = ser.read(9)
            ser.reset_input_buffer()
            print("Analyze Road Condition")
            time.sleep(0.1)

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: # this portion is for python3          
                distance = bytes_serial[2] + bytes_serial[3]*256 # multiplied by 256, because the binary data is shifted by 8 to the left (equivalent to "<< 8").                                              # Dist_L, could simply be added resulting in 16-bit data of Dist_Total.
                
                if distance == 3:
                    print("This is normal")
                elif distance > 3:
                    print("Pothole")
                    path = 'alarm.wav'
                    pygame.mixer.init()
                    speaker_volume = 0.5
                    pygame.mixer.music.set_volume(speaker_volume)
                    pygame.mixer.music.load(path)
                        
                    pygame.mixer.music.play()
                else:
                    print("Bonggol")
                time.sleep(3)
                ser.reset_input_buffer()


if __name__ == "__main__":
    try:
        if ser.isOpen() == False:
            ser.open()
        read_data()
    except KeyboardInterrupt(): # ctrl + c in terminal.
        if ser != None:
            ser.close()
            print("program interrupted by the user")



