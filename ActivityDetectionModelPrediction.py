import pymoduleconnector
from pymoduleconnector import DataType
from pymoduleconnector import RecordingOptions
import time
import numpy as np
from vmdpy import VMD  
from sklearn.decomposition import PCA
import math
from scipy.signal import sosfiltfilt, butter,filtfilt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from drawnow import drawnow, figure
from joblib import load
#from ModelPrediction import model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model




# Loading the SVM model
# svm_model = load(r'C:\Users\2680850K\Desktop\XeThru Recordings\svm_model.joblib')

# # Loading the Random Forest model
# rf_model = load(r'C:\Users\2680850K\Desktop\XeThru Recordings\rf_model.joblib')

def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    time.sleep(3)


global predictionPlot 
predictionPlot = 0

def main():
    device_name = "COM5"
    enable_recording = 1
    output_dir = 'C:/Users/2680850K/Desktop/XeThru Recordings/'
    
  

    pca = PCA()

    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    print("receive ping, the value is:", hex(xep.ping()))
    fwid = xep.get_system_info(2)

    session_id='liam'


    # # Display system info
    print ('FirmWareID = ', fwid)
    print ('Version = ', xep.get_system_info(3))
    print ('Build = ', xep.get_system_info(4))
    print ('SerialNumber = ', xep.get_system_info(6))
    print ('VersionList = ', xep.get_system_info(7))
    fps = 50

    xep.x4driver_init()

    xep.x4driver_set_downconversion(1)

    xep.x4driver_set_iterations(16)
    xep.x4driver_set_pulses_per_step(10)
    xep.x4driver_set_dac_min(949)
    xep.x4driver_set_dac_max(1100)

    xep.x4driver_set_frame_area_offset(0.18)
    xep.x4driver_set_frame_area(0.4, 5.0)

    frames = xep.x4driver_get_frame_area()

    frame_start = frames.start
    frame_stop = frames.end

    if enable_recording:
        recorder = mc.get_data_recorder()
        RecordingOptions1 = RecordingOptions()
        RecordingOptions1.set_session_id(session_id)
        duration = pymoduleconnector.moduleconnectorwrapper.PreferredSplitSize()
        duration.set_duration(10)
        baseband_ap_type = (DataType.FloatDataType)
        recorder.start_recording(baseband_ap_type,output_dir)

    xep.x4driver_set_fps(fps)
    infocache = np.empty((0,0))
    #Infocache = []
    print (infocache.shape)


    xep.read_message_data_float()
    x = 0
    #new code
    AmpReading = []
    PhaseReading = []
    # Initialize plot
    # plt.ion()  # Enable interactive mode
    # fig, ax = plt.subplots(figsize=(10, 4))
    # line, = ax.plot(AmpReading)
    # plt.title('Average Amplitude of IQ Data Over Time')
    # plt.xlabel('Time')
    # plt.ylabel('Average Amplitude')
    # plt.grid(True)
    
    # Initialize plot
    
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots

    # Amplitude plot
    line1, = ax1.plot(AmpReading)
    ax1.set_title('Average Amplitude of IQ Data Over Time ')
    ax1.set_ylabel('Average Amplitude')
    ax1.grid(True)

    # Phase plot
    line2, = ax2.plot(PhaseReading)
    ax2.set_title('Average Phase of IQ Data Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Phase (radians)')
    ax2.grid(True)
     
    while True:
        contentID = xep.read_message_data_float().content_id
        info = xep.read_message_data_float().info
        data=np.asarray(xep.read_message_data_float().data)
        data_length = len(data)
        #print('data length',data_length)


        # Generate IQ vector.
        i_vec = data[0:int(data_length/2)]
        q_vec = data[int(data_length/2):]
        iq_vec = i_vec + 1j * q_vec
        #print('IQ vector ',len(iq_vec))
        iq_vec = iq_vec.reshape(iq_vec.shape[0],1)
        amplitude = np.abs(iq_vec)  # Calculate the amplitude of the IQ vector        
        
        
        #new code
        #avg_amplitude = np.mean(amplitude, axis=0)  # Averaging over the second dimension
        # Ravel the array
        avg_amplitude = amplitude.ravel()
        print('length ',len(avg_amplitude))
        print('shape',avg_amplitude.shape)
        #AmpReading.append(avg_amplitude)  # Store the average amplitude
        for value in avg_amplitude:
            AmpReading.append(value)
        
        #Model prediction 
        #if len(AmpReading) >= 91 and len(AmpReading) % 91 == 0:
        # Extract the last 91 elements
        #input_data = np.array(AmpReading[-91:]).reshape(1, -1)
        input_data = np.array(avg_amplitude).reshape(1, -1)
        model = load_model(r'C:\Users\2680850K\Desktop\XeThru Recordings\model_8January.h5')
            #model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logits = model.predict(input_data)
        predicted_class = tf.argmax(logits, axis=1).numpy()[0]
        predictionPlot  = predicted_class    
        ax1.set_title(f'Average Amplitude of IQ Data Over Time => Class Prediction {predictionPlot}')
            #print("Predicted class:", predicted_class)

        phase = np.angle(iq_vec)    # Calculate the phase of the IQ vector
        avg_phase = np.mean(phase, axis=0)
        PhaseReading.append(avg_phase.item())

        # Update amplitude plot
        line1.set_ydata(AmpReading)
        line1.set_xdata(range(len(AmpReading)))
        ax1.relim()
        ax1.autoscale_view(True,True,True)

        # Update phase plot
        line2.set_ydata(PhaseReading)
        line2.set_xdata(range(len(PhaseReading)))
        ax2.relim()
        ax2.autoscale_view(True,True,True)
        

        plt.draw()
        plt.pause(0.1)  # Pause briefly for the plot to update
      
        threshold = 10

        #output_resp,output_heart,infocache = signal_update(iq_vec, infocache,91,threshold,fps,pca)


        x = x + 1
    

if __name__ == "__main__":
    main()


        
