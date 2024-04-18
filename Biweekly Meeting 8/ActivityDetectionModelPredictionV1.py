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

# Initializing global variables
global predictionPlot 
predictionPlot = 0

def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    time.sleep(3)

def main():
    device_name = "COM5"
    enable_recording = 1
    output_dir = 'C:/Users/2680850K/Desktop/XeThru Recordings/'

    # Initialize PCA and other necessary components
    pca = PCA()

    # Establish connection to XeThru radar
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
    fps = 25

    xep.x4driver_init()

    xep.x4driver_set_downconversion(1)

    xep.x4driver_set_iterations(16)
    xep.x4driver_set_pulses_per_step(100)
    xep.x4driver_set_dac_min(949)
    xep.x4driver_set_dac_max(1100)

    xep.x4driver_set_frame_area_offset(0.18)
    xep.x4driver_set_frame_area(0.4, 5.0)

    frames = xep.x4driver_get_frame_area()

    frame_start = frames.start
    frame_stop = frames.end


    # Start recording if enabled
    if enable_recording:
        recorder = mc.get_data_recorder()
        RecordingOptions1 = RecordingOptions()
        RecordingOptions1.set_session_id('liam')
        duration = pymoduleconnector.moduleconnectorwrapper.PreferredSplitSize()
        duration.set_duration(10)
        baseband_ap_type = DataType.FloatDataType
        recorder.start_recording(baseband_ap_type, output_dir)
    
    xep.x4driver_set_fps(fps)
    infocache = np.empty((0,0))
    #Infocache = []
    print (infocache.shape)

    # Set up the plot for displaying data
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    line1, = ax1.plot([])
    ax1.set_title('Average Amplitude of IQ Data Over Time')
    ax1.set_ylabel('Average Amplitude')
    ax1.grid(True)

    line2, = ax2.plot([])
    ax2.set_title('Average Phase of IQ Data Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Phase (radians)')
    ax2.grid(True)

    # Initialize data lists
    AmpReading = []
    PhaseReading = []

    while True:
        # Read data from XeThru radar
        contentID = xep.read_message_data_float().content_id
        info = xep.read_message_data_float().info
        data = np.asarray(xep.read_message_data_float().data)
        data_length = len(data)

        # Process IQ data and calculate amplitude and phase
        i_vec = data[0:int(data_length/2)]
        q_vec = data[int(data_length/2):]
        iq_vec = i_vec + 1j * q_vec
        amplitude = np.abs(iq_vec)
        phase = np.angle(iq_vec)

        # Store amplitude and phase readings
        AmpReading.extend(amplitude.ravel())
        PhaseReading.append(np.mean(phase))

        # Make predictions using a sliding window approach
        if len(AmpReading) >= 91:  # Assuming a window size of 91
            input_data = np.array(AmpReading[-91:]).reshape(1, -1)
            model = load_model(r'C:\Users\2680850K\Desktop\XeThru Recordings\model.h5')
            logits = model.predict(input_data)
            predicted_class = tf.argmax(logits, axis=1).numpy()[0]
            predictionPlot = predicted_class
            ax1.set_title(f'Average Amplitude of IQ Data => Class Prediction: {predictionPlot}')

        # Update plot with new data
        line1.set_ydata(AmpReading)
        line1.set_xdata(range(len(AmpReading)))
        ax1.relim()
        ax1.autoscale_view(True, True, True)

        line2.set_ydata(PhaseReading)
        line2.set_xdata(range(len(PhaseReading)))
        ax2.relim()
        ax2.autoscale_view(True, True, True)

        plt.draw()
        plt.pause(0.1)

if __name__ == "__main__":
    main()
