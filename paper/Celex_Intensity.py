import os,sys
import numpy as np
import math 
from cv2 import cv2 as cv
from Celex_Video import IntensityVideo
import time 



## Framing based on the latest updata of every pixel's illumination intensity within a timewindow sampling

# with denoising
def Load_Event1(filepath,savedir,BytesPerEvent,TimeWindow,Brightness_map,heatMap,bin_order,mode,heatMap_counter,pic_seq,y):

    Y_pixel_total = 640
    X_pixel_total = 768
    '''
    upside-down operation in terms of y-axis
    and some negative values occur, still confused about it which is derived from CelexMatlabToolBox 
    ''' 
    with open(filepath,'rb') as File:
        if bin_order == 1: 
            j = 1
            File.seek(1,0)
            count = 0
            while j>0:
                data = File.read(BytesPerEvent)
                count = count + 1 
                if (data[3]>0) & (data[3] !=255) == 1:
                    File.seek(4*(count-1),0) # pointer move back to the first byte of the first row event data
                    break
            bin_order = 0
        
        Bytes_Index = 0
        Raw_Event = File.read()
        bytes_Recorded = len(Raw_Event)
        # print('the length of bin file is %d'%bytes_Recorded)


        # Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        # heatmap = np.zeros((Y_pixel_total,X_pixel_total))  # 记录每一个frame内的heat情况


        Sample_count = 0
        i = pic_seq
        # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % (TimeWindow * BytesPerEvent))):
        try:
            # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % BytesPerEvent)):
            while Bytes_Index < bytes_Recorded:
                # print('Event_count = %d'%Event_count)
                Byte_1 = Raw_Event[Bytes_Index]
                Byte_2 = Raw_Event[Bytes_Index+1]
                Byte_3 = Raw_Event[Bytes_Index+2]
                Byte_4 = Raw_Event[Bytes_Index+3]
                # print('Byte_4 = %d'%Byte_4)
                if Byte_4 == 0:
                    x_lower = Byte_1 % 2**7    #x[6:0]
                    x_higher = (Byte_2 % 2**6) - (Byte_2 % 2**4) #x[8,7]
                    x = (x_higher << 3) + x_lower

                    adc_lower = Byte_2 % 2**4 #ADC[3:0]
                    adc_upper = Byte_3 % 2**5 #ADC[8:4]
                    adc = (adc_upper << 4) + adc_lower

                    C = (Byte_2 >> 6)
                    if C == 1:
                        x = int(767-x)

                    if x<0:
                        event_x = 0
                    else:
                        event_x =int(x)
                    if y>639:
                        event_y = int(639)
                    else:
                        event_y = int(639-y)

                    # print(type(Brightness_map))
                    Brightness_map[event_y,event_x] = adc

                    heatMap[event_y,event_x] = heatMap[event_y,event_x]+1

                    Bytes_Index = Bytes_Index + 4
                    Sample_count = Sample_count + 1

                    # print('Sample)count is %d'%Sample_count)

                    if Sample_count == TimeWindow:            
                        thresh=TimeWindow /sum(sum(heatMap>0))
                        if  thresh <= 2:
                            thresh=2
                        else:
                            thresh=math.floor(thresh)

                        # # thresh = 2
                        weightMap = (heatMap >= thresh) + 0
                        ## Method-1
                        Brightness_temp = 255*(Brightness_map - np.min(Brightness_map))/(np.max(Brightness_map)-np.min(Brightness_map))
                        ## Method-2
                        # Brightness_temp = 255*(Brightness_map - 0)/(511. - 0.)
                        ## method-5
                        # Brightness_temp = (Brightness_map - 0)/(511.-0.) + 127/255.
                        # Brightness_temp = Brightness_temp * 255


                        i = i+1
                        Sample_count = 0
                        # fimg = Brightness_temp
                        fimg = Brightness_temp * weightMap
                        cv.imwrite(savedir+str(i)+'.jpg',fimg)
                        # print('New pic:',savedir+str(i)+'.jpg')


                        # cv.imshow("Intensity_graph",fimg)
                        # cv.resizeWindow("Intensity_graph", X_pixel_total, Y_pixel_total)
                        # cv.waitKey(70)


                        ## the END of a intensity image framing
                        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))

                        # heatMap = np.zeros((Y_pixel_total,X_pixel_total))     
                        heatMap_counter = heatMap_counter + 1                   #按照一定取样间隔来重置heatMap （此处为生成50帧进行清零）
                        if ((heatMap_counter % 50)==0 ):
                            heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                    

                        # method-3 : interpolate the heatmap by 50 frames
                        # if (heatMap_counter > 50):
                        #     heatMap_temp = heatmap
                        # if (heatMap_counter > 50) & ((heatMap_counter % 50) == 0): # 覆盖50个frame的信息进行heatMap的更新
                        #     heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                        #     heatMap = heatMap_temp
                        # if ((heatMap_counter - 50) % 50) == 0:
                        #     heatMap_temp = np.zeros((Y_pixel_total,X_pixel_total))
                        

                elif Byte_4 == 255:
                    Bytes_Index = Bytes_Index + 4
                else:
                    y_lower = Byte_1 % 2**7 #Y[6:0]
                    y_higher = (Byte_2 % 2**7 - Byte_2 % 2**4) #Y[9,7]
                    y = (y_higher << 3) + y_lower
                    Bytes_Index = Bytes_Index + 4
        except:
            print('Not enough events!')

        # print('Bytes_Index is %d'%Bytes_Index)
        # print('There are %d events generated!!'%Event_count)
        # print('And there are %d events of special type'%count_special)
        # name = os.path.split(filepath)[1]
        # print('The events extracted from',name,'!\n')

    if mode == 'Consecutive':
        pic_seq = i
        ## mode-1:
        return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y,i
        # ## mode-4
        # return Brightness_map

    elif mode == 'Non_Consecutive':
        bin_order = 1
        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap_counter = 0
        y = 0
        print('One bin file successfully extracted !!') 
        # return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y
        return i

    else:
        print('framing mode error!!!')
        sys.exit()

# without denoising
def Load_Event2(filepath,savedir,BytesPerEvent,TimeWindow,Brightness_map,heatMap,bin_order,mode,heatMap_counter,pic_seq,y):

    Y_pixel_total = 640
    X_pixel_total = 768
    '''
    upside-down operation in terms of y-axis
    and some negative values occur, still confused about it which is derived from CelexMatlabToolBox 
    ''' 
    with open(filepath,'rb') as File:
        if bin_order == 1: 
            j = 1
            File.seek(1,0)
            count = 0
            while j>0:
                data = File.read(BytesPerEvent)
                count = count + 1 
                if (data[3]>0) & (data[3] !=255) == 1:
                    File.seek(4*(count-1),0) # pointer move back to the first byte of the first row event data
                    break
            bin_order = 0
        
        Bytes_Index = 0
        Raw_Event = File.read()
        bytes_Recorded = len(Raw_Event)
        # print('the length of bin file is %d'%bytes_Recorded)


        # Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        # heatmap = np.zeros((Y_pixel_total,X_pixel_total))  # 记录每一个frame内的heat情况


        Sample_count = 0
        i = pic_seq
        # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % (TimeWindow * BytesPerEvent))):
        try:
            # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % BytesPerEvent)):
            while Bytes_Index < bytes_Recorded:
                # print('Event_count = %d'%Event_count)
                Byte_1 = Raw_Event[Bytes_Index]
                Byte_2 = Raw_Event[Bytes_Index+1]
                Byte_3 = Raw_Event[Bytes_Index+2]
                Byte_4 = Raw_Event[Bytes_Index+3]
                # print('Byte_4 = %d'%Byte_4)
                if Byte_4 == 0:
                    x_lower = Byte_1 % 2**7    #x[6:0]
                    x_higher = (Byte_2 % 2**6) - (Byte_2 % 2**4) #x[8,7]
                    x = (x_higher << 3) + x_lower

                    adc_lower = Byte_2 % 2**4 #ADC[3:0]
                    adc_upper = Byte_3 % 2**5 #ADC[8:4]
                    adc = (adc_upper << 4) + adc_lower

                    C = (Byte_2 >> 6)
                    if C == 1:
                        x = int(767-x)

                    if x<0:
                        event_x = 0
                    else:
                        event_x =int(x)
                    if y>639:
                        event_y = int(639)
                    else:
                        event_y = int(639-y)

                    # print(type(Brightness_map))
                    Brightness_map[event_y,event_x] = adc

                    # heatMap[event_y,event_x] = heatMap[event_y,event_x]+1

                    Bytes_Index = Bytes_Index + 4
                    Sample_count = Sample_count + 1

                    # print('Sample)count is %d'%Sample_count)

                    if Sample_count == TimeWindow:
                        # thresh=TimeWindow /sum(sum(heatMap>0))
                        # if  thresh <= 2:
                        #     thresh=2
                        # else:
                        #     thresh=math.floor(thresh)

                        # # # thresh = 2
                        # weightMap = (heatMap >= thresh) + 0
                        ## Method-1
                        Brightness_temp = 255*(Brightness_map - np.min(Brightness_map))/(np.max(Brightness_map)-np.min(Brightness_map))
                       
                        ## Method-2
                        # Brightness_temp = 255*(Brightness_map - 0)/(511. - 0.)
                        

                        i = i+1
                        Sample_count = 0
                        fimg = Brightness_temp

                        # fimg = Brightness_temp * weightMap
                        cv.imwrite(savedir+str(i)+'.jpg',fimg)
                        # print('New pic:',savedir+str(i)+'.jpg')


                        # the end of a intensity framing
                        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))

    

                elif Byte_4 == 255:
                    Bytes_Index = Bytes_Index + 4
                else:
                    y_lower = Byte_1 % 2**7 #Y[6:0]
                    y_higher = (Byte_2 % 2**7 - Byte_2 % 2**4) #Y[9,7]
                    y = (y_higher << 3) + y_lower
                    Bytes_Index = Bytes_Index + 4
        except:
            print('Not enough events!')

    if mode == 'Consecutive':
        pic_seq = i
        ## mode-1:
        return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y,i
        # ## mode-4
        # return Brightness_map
    elif mode == 'Non_Consecutive':
        bin_order = 1
        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap_counter = 0
        y = 0
        print('One bin file successfully extracted !!') 
        # return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y
        return i
    else:
        print('framing mode error!!!')
        sys.exit()



## Framing based on the 255 slices within a timewindow sampling 

# with denoising
def Framing_slice1(filepath,savedir,BytesPerEvent,slice_size,Brightness_map,heatMap,bin_order,mode,heatMap_counter,pic_seq,y):
    # global brightness_map
    # global heat_map
    # global heat_map_counter

    TimeWindow = 255*slice_size        
    Y_pixel_total = 640
    X_pixel_total = 768
    '''
    upside-down operation in terms of y-axis
    and some negative values occur, still confused about it which is derived from CelexMatlabToolBox 
    ''' 
    with open(filepath,'rb') as File:
        if bin_order == 1: 
            j = 1
            File.seek(1,0)
            count = 0
            while j>0:
                data = File.read(BytesPerEvent)
                count = count + 1 
                # print('the count is%d'%count)
                if (data[3]>0) & (data[3] !=255) == 1:
                    File.seek(4*(count-1),0) # pointer move back to the first byte of the first row event data
                    break
            bin_order = 0
        
        Bytes_Index = 0
        Raw_Event = File.read()
        bytes_Recorded = len(Raw_Event)
        # print('the length of bin file is %d'%bytes_Recorded)

        # Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        # heatMap = np.zeros((Y_pixel_total,X_pixel_total))

        # heatmap = np.zeros((Y_pixel_total,X_pixel_total))  # 记录每一个frame内的heat情况

        Sample_count = 0
        i = pic_seq
        slice_num = 1
        # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % (TimeWindow * BytesPerEvent))):
        try:
            # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % BytesPerEvent)):
            while Bytes_Index < bytes_Recorded:
                # print('Event_count = %d'%Event_count)
                Byte_1 = Raw_Event[Bytes_Index]
                Byte_2 = Raw_Event[Bytes_Index+1]
                # Byte_3 = Raw_Event[Bytes_Index+2]
                Byte_4 = Raw_Event[Bytes_Index+3]
                # print('Byte_4 = %d'%Byte_4)
                if Byte_4 == 0:
                    x_lower = Byte_1 % 2**7    #x[6:0]
                    x_higher = (Byte_2 % 2**6) - (Byte_2 % 2**4) #x[8,7]
                    x = (x_higher << 3) + x_lower

                    # print('x is detected')
                    # adc_lower = Byte_2 % 2**4 #ADC[3:0]
                    # adc_upper = Byte_3 % 2**5 #ADC[8:4]
                    # adc = (adc_upper << 4) + adc_lower

                    C = (Byte_2 >> 6)
                    if C == 1:
                        x = int(767-x)

                    if x<0:
                        event_x = 0
                    else:
                        event_x =int(x)
                    if y>639:
                        event_y = int(639)
                    else:
                        event_y = int(639-y)

                    # print(type(Brightness_map))
                    Brightness_map[event_y,event_x] = slice_num

                    heatMap[event_y,event_x] = heatMap[event_y,event_x]+1

                    Bytes_Index = Bytes_Index + 4
                    Sample_count = Sample_count + 1

                    if (Sample_count % slice_size) == 0:
                        slice_num = slice_num + 1

                    # print('Sample)count is %d'%Sample_count)

                    if Sample_count == TimeWindow:
                        thresh=TimeWindow /sum(sum(heatMap>0))
                        if  thresh <= 2:
                            thresh=2
                        else:
                            thresh=math.floor(thresh)

                        # thresh = 2
                        weightMap = (heatMap >= thresh) + 0

                        ## Method-1    Max-Min normalization
                        Brightness_temp = 255*(Brightness_map - np.min(Brightness_map))/(np.max(Brightness_map)-np.min(Brightness_map))
                        
                        ## Method-2
                        # Brightness_temp = 255*(Brightness_map - 0)/(511. - 0.)
                        ## method-5
                        # Brightness_temp = (Brightness_map - 0)/(511.-0.) + 127/255.
                        # Brightness_temp = Brightness_temp * 255

                        i = i+1
                        Sample_count = 0
                        slice_num = 1
                        # fimg = Brightness_map
                        fimg = Brightness_temp * weightMap
                        # print(savedir+str(i)+'.jpg')
                        cv.imwrite(savedir+str(i)+'.jpg',fimg)
                        # print('New pic:',savedir+str(i)+'.jpg')

                        # cv.imshow("Intensity_graph",fimg)
                        # cv.resizeWindow("Intensity_graph", X_pixel_total, Y_pixel_total)
                        # cv.waitKey(70)

                        # the end of a intensity framing
                        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))

                        ## heatMap = np.zeros((Y_pixel_total,X_pixel_total))

                        heatMap_counter = heatMap_counter + 1
                        if ((heatMap_counter % 50)==0 ):
                            heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                    
                        # method-3 : interpolate the heatmap by 50 frames
                        # if (heatMap_counter > 50):
                        #     heatMap_temp = heatmap
                        # if (heatMap_counter > 50) & ((heatMap_counter % 50) == 0): # 覆盖50个frame的信息进行heatMap的更新
                        #     heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                        #     heatMap = heatMap_temp
                        # if ((heatMap_counter - 50) % 50) == 0:
                        #     heatMap_temp = np.zeros((Y_pixel_total,X_pixel_total))
                        

                elif Byte_4 == 255:
                    Bytes_Index = Bytes_Index + 4
                else:
                    y_lower = Byte_1 % 2**7 #Y[6:0]
                    y_higher = (Byte_2 % 2**7 - Byte_2 % 2**4) #Y[9,7]
                    y = (y_higher << 3) + y_lower
                    Bytes_Index = Bytes_Index + 4
        except:
            print('Not enough events!')

        # print('Bytes_Index is %d'%Bytes_Index)
        
        # print('There are %d events generated!!'%Event_count)
        # print('And there are %d events of special type'%count_special)
        # name = os.path.split(filepath)[1]
        # print('The events extracted from',name,'!\n')
    if mode == 'Consecutive':
        pic_seq = i
        ## mode-1:
        return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y,i
        # ## mode-4
        # return Brightness_map
    elif mode == 'Non_Consecutive':
        bin_order = 1
        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap_counter = 0
        y = 0
        print('One bin file successfully extracted !!') 
        # return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y
        return i
    else:
        print('framing mode error!!!')
        sys.exit()


# without denoising
def Framing_slice2(filepath,savedir,BytesPerEvent,slice_size,Brightness_map,heatMap,bin_order,mode,heatMap_counter,pic_seq,y):
    # global brightness_map
    # global heat_map
    # global heat_map_counter

    TimeWindow = 255*slice_size        
    Y_pixel_total = 640
    X_pixel_total = 768
    '''
    upside-down operation in terms of y-axis
    and some negative values occur, still confused about it which is derived from CelexMatlabToolBox 
    ''' 
    with open(filepath,'rb') as File:
        if bin_order == 1: 
            j = 1
            File.seek(1,0)
            count = 0
            while j>0:
                data = File.read(BytesPerEvent)
                count = count + 1 
                # print('the count is%d'%count)
                if (data[3]>0) & (data[3] !=255) == 1:
                    File.seek(4*(count-1),0) # pointer move back to the first byte of the first row event data
                    break
            bin_order = 0
        
        Bytes_Index = 0
        Raw_Event = File.read()
        bytes_Recorded = len(Raw_Event)
        # print('the length of bin file is %d'%bytes_Recorded)

        # Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        # heatMap = np.zeros((Y_pixel_total,X_pixel_total))

        # heatmap = np.zeros((Y_pixel_total,X_pixel_total))  # 记录每一个frame内的heat情况

        Sample_count = 0
        i = pic_seq
        slice_num = 1
        # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % (TimeWindow * BytesPerEvent))):
        try:
            # while Bytes_Index < (bytes_Recorded - (bytes_Recorded % BytesPerEvent)):
            while Bytes_Index < bytes_Recorded:
                # print('Event_count = %d'%Event_count)
                Byte_1 = Raw_Event[Bytes_Index]
                Byte_2 = Raw_Event[Bytes_Index+1]
                # Byte_3 = Raw_Event[Bytes_Index+2]
                Byte_4 = Raw_Event[Bytes_Index+3]
                # print('Byte_4 = %d'%Byte_4)
                if Byte_4 == 0:
                    x_lower = Byte_1 % 2**7    #x[6:0]
                    x_higher = (Byte_2 % 2**6) - (Byte_2 % 2**4) #x[8,7]
                    x = (x_higher << 3) + x_lower

                    # print('x is detected')
                    # adc_lower = Byte_2 % 2**4 #ADC[3:0]
                    # adc_upper = Byte_3 % 2**5 #ADC[8:4]
                    # adc = (adc_upper << 4) + adc_lower

                    C = (Byte_2 >> 6)
                    if C == 1:
                        x = int(767-x)

                    if x<0:
                        event_x = 0
                    else:
                        event_x =int(x)
                    if y>639:
                        event_y = int(639)
                    else:
                        event_y = int(639-y)

                    # print(type(Brightness_map))
                    Brightness_map[event_y,event_x] = slice_num

                    heatMap[event_y,event_x] = heatMap[event_y,event_x]+1

                    Bytes_Index = Bytes_Index + 4
                    Sample_count = Sample_count + 1

                    if (Sample_count % slice_size) == 0:
                        slice_num = slice_num + 1

                    # print('Sample)count is %d'%Sample_count)

                    if Sample_count == TimeWindow:
                        # thresh=TimeWindow /sum(sum(heatMap>0))
                        # if  thresh <= 2:
                        #     thresh=2
                        # else:
                        #     thresh=math.floor(thresh)

                        # # thresh = 2
                        # weightMap = (heatMap >= thresh) + 0

                        ## Method-1
                        # Brightness_temp = 255*(Brightness_map - np.min(Brightness_map))/(np.max(Brightness_map)-np.min(Brightness_map))
                        
                        ## Method-2
                        # Brightness_temp = 255*(Brightness_map - 0)/(511. - 0.)
                        ## method-5
                        # Brightness_temp = (Brightness_map - 0)/(511.-0.) + 127/255.
                        # Brightness_temp = Brightness_temp * 255

                        i = i+1
                        Sample_count = 0
                        slice_num = 1
                        fimg = Brightness_map
                        # fimg = Brightness_temp * weightMap
                        cv.imwrite(savedir+str(i)+'.jpg',fimg)
                        # print('New pic:',savedir+str(i)+'.jpg')

                        # cv.imshow("Intensity_graph",fimg)
                        # cv.resizeWindow("Intensity_graph", X_pixel_total, Y_pixel_total)
                        # cv.waitKey(70)

                        # the end of a intensity framing
                        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))

                        ## heatMap = np.zeros((Y_pixel_total,X_pixel_total))

                        # heatMap_counter = heatMap_counter + 1
                        # if ((heatMap_counter % 50)==0 ):
                        #     heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                    
                        # method-3 : interpolate the heatmap by 50 frames
                        # if (heatMap_counter > 50):
                        #     heatMap_temp = heatmap
                        # if (heatMap_counter > 50) & ((heatMap_counter % 50) == 0): # 覆盖50个frame的信息进行heatMap的更新
                        #     heatMap = np.zeros((Y_pixel_total,X_pixel_total))
                        #     heatMap = heatMap_temp
                        # if ((heatMap_counter - 50) % 50) == 0:
                        #     heatMap_temp = np.zeros((Y_pixel_total,X_pixel_total))
                        

                elif Byte_4 == 255:
                    Bytes_Index = Bytes_Index + 4
                else:
                    y_lower = Byte_1 % 2**7 #Y[6:0]
                    y_higher = (Byte_2 % 2**7 - Byte_2 % 2**4) #Y[9,7]
                    y = (y_higher << 3) + y_lower
                    Bytes_Index = Bytes_Index + 4
        except:
            print('Not enough events!')

        # print('Bytes_Index is %d'%Bytes_Index)
        
        # print('There are %d events generated!!'%Event_count)
        # print('And there are %d events of special type'%count_special)
        # name = os.path.split(filepath)[1]
        # print('The events extracted from',name,'!\n')
    if mode == 'Consecutive':
        pic_seq = i
        ## mode-1:
        return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y,i
        # ## mode-4
        # return Brightness_map
    elif mode == 'Non_Consecutive':
        bin_order = 1
        Brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap = np.zeros((Y_pixel_total,X_pixel_total))
        heatMap_counter = 0
        y = 0
        print('One bin file successfully extracted !!') 
        # return Brightness_map, heatMap, pic_seq, bin_order, heatMap_counter, y
        return i
    else:
        print('framing mode error!!!')
        sys.exit()





if __name__ == "__main__":
    # heat_map_counter = 0
    # Y_pixel_total = 640
    # X_pixel_total = 768
    # brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
    # heat_map = np.zeros((Y_pixel_total,X_pixel_total))
    # Bin_order = 0
    # y = 0

    heat_map_counter = 0
    Y_pixel_total = 640
    X_pixel_total = 768
    brightness_map = np.zeros((Y_pixel_total,X_pixel_total))
    heat_map = np.zeros((Y_pixel_total,X_pixel_total))
    Bin_order = 1
    y_ini = 0 
    seq = 0
    file_dir = 'O:/Celex_code_PY/Celex_Preprocessing/Celex_Camera/'
    savedir = 'O:/Celex_code_PY/Celex_Preprocessing/Celex_Displaying/'
    Video_dir = 'O:/Celex_code_PY/Celex_RealTime/RealTime_video/'

    for filepath in os.listdir(file_dir):
        file_name = os.path.splitext(filepath)[0]
        start_time = time.time()
        
        # pic_num = Load_Event1(file_dir + filepath,savedir,4,13260,brightness_map,heat_map,Bin_order,'Non_Consecutive',heat_map_counter,seq,y_ini)
        # pic_num = Load_Event1(file_dir + filepath,savedir,4,5000,brightness_map,heat_map,Bin_order,'Consecutive',heat_map_counter,seq,y_ini)
        # pic_num = Load_Event2(file_dir + filepath,savedir,4,10000,brightness_map,heat_map,Bin_order,'Consecutive',heat_map_counter,seq,y_ini)
        
        # pic_num = Framing_slice1(file_dir + filepath,savedir,4,39,brightness_map,heat_map,Bin_order,'Non_Consecutive',heat_map_counter,seq,y_ini)
        # pic_num = Framing_slice1(file_dir + filepath,savedir,4,39,brightness_map,heat_map,Bin_order,'Consecutive',heat_map_counter,seq,y_ini)
        # pic_num = Framing_slice2(file_dir + filepath,savedir,4,39,brightness_map,heat_map,Bin_order,'Consecutive',heat_map_counter,seq,y_ini)

        end_time= time.time()
        Time = end_time - start_time
        # print('the framing time is %d s for %d pics'%(Time,pic_num[6]))

        IntensityVideo(savedir,file_name,Video_dir,30,(768,640))

        # print('deleting the pics now...')
        # for file in os.listdir(savedir):
        #         os.unlink(os.path.join(savedir+file))

    print('Process finished !!')


