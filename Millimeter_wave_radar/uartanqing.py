"""
这个文件是丁建新师兄发来的，处理毫米波雷达信息的文件，现在先试图看懂
可以尝试更改一下
"""

import serial
import tkinter
import os
import tkinter
import tkinter.messagebox
import tkinter.filedialog
import time
import struct
import math
import serial
import json

def ByteToHex( bins ):
    return ''.join( [ "%02X" % x for x in bins ] ).strip()


if __name__ == "__main__":

    ser=serial.Serial('/dev/ttyACM0',115200,timeout=1)
    a=ser.isOpen()
    # print(a)
    #ser.write("hello".encode())
    # path = tkinter.filedialog.askopenfilename()
    # print(path)
    path='test.txt'
    f = open(path, 'r',encoding='UTF-8')
    i=1
    while i<45:
        txt = f.readline()
        print(txt)
        time.sleep(0.2)
        ser.write(txt.encode())
        i=i+1
    f.close()
    ser.close()
    ser1=serial.Serial('/dev/ttyACM1',921600,timeout=0.002,inter_byte_timeout=0.0001)
    ser1.stopbits=1
    ser1.bytesize=8
    # c=ser1.isOpen()
    # print(c)
    ser1.close()
    ser1.open()
    ser1.write("hello".encode())

    while(1):
        # data = ser1.readline(100)
        # #data1=bytes.fromhex(data)
        # #hexShow(data)
        # print(data)
        # print(type(data))
        # ser1.flushInput()
        # ser1.flush()
        ser1.flush()
        temp =ser1.read_all()

        # flush()
        # print(temp)
        # time.sleep(2)
        # print(temp[0])
        temp = ByteToHex(temp)
        # index=str.find("0201040306050807")
        # print(temp)
        array=list(temp)#接收数据存成list
        e= array[96:100]
        # print(e)
        # int(ff, 16) #16进制转换为10进制
        range1_list =[]
        azimuth_list=[]
        elevation_list=[]
        doppler_list=[]
        if temp[0:16]=="0201040306050807":
            length=int(array[112],16)*16+int(array[113],16)+int(array[114],16)*16**3+int(array[115],16)*16**2
            length1 = int(array[114]+array[115]+array[112]+array[113], 16)
            # print(length1)
            # print(length)
            if temp[104:108]=="0600":
                numpoint=(length1-16)/32
                numpoint = 1
                i=1
                while i <=numpoint:
                    index=32*(i-1)+120
                    # range=int(array[index+6]+array[index+7]+array[index+4]+array[index+5]+array[index+2]+array[index+3]+array[index]+array[index+1],16)
                    range1 =array[index+6]+array[index+7]+array[index+4]+array[index+5]+array[index+2]+array[index+3]+array[index]+array[index+1]
                    range1_f = struct.unpack('!f', bytes.fromhex(range1))[0]
                    # print(range1_f)
                    range1_list.append(range1_f)
                    index = index+8
                    azimuth=array[index+6]+array[index+7]+array[index+4]+array[index+5]+array[index+2]+array[index+3]+array[index]+array[index+1]
                    azimuth_f = struct.unpack('!f', bytes.fromhex(azimuth))[0]
                    # print(azimuth_f)
                    azimuth_list.append(azimuth_f)
                    index = index + 8
                    elevation = array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5] + array[index + 2] + array[index + 3] + array[index] + array[index + 1]
                    elevation_f = struct.unpack('!f', bytes.fromhex(elevation))[0]
                    # print(elevation_f)
                    elevation_list.append(elevation_f)
                    index = index + 8
                    doppler = array[index + 6] + array[index + 7] + array[index + 4] + array[index + 5] + array[index + 2] + array[index + 3] + array[index] + array[index + 1]
                    doppler_f = struct.unpack('!f', bytes.fromhex(doppler))[0]
                    # print(doppler_f)
                    doppler_list.append(doppler_f)
                    i=i+1
                x = range1_f * math.sin(azimuth_f)
                y = range1_f * math.cos(azimuth_f)
                z = range1_f * math.sin(elevation_f)
                efflist="%f %f %f" %(x,y,z)
                print(efflist)
                post_data = {"Obstacle":
                    {
                    "Obstacle_X": float(x),
                    "Obstacle_Y": float(y),
                    "Obstacle_Z": float(z),
                 }
                }
                print(json.dumps(post_data))
                #os.system(
                # f"python3 drone_sdk.py -f linkkit_thing_post_property -j '{json.dumps(post_data)}' -s 0")
                time.sleep(0.5)

  
          
          
           
              


