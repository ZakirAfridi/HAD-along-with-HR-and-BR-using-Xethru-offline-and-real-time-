import pymoduleconnector
from pymoduleconnector import DataType, RecordingOptions
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
import tkinter as tk
from PIL import Image, ImageTk
from pymoduleconnector import DataType, RecordingOptions


# Initializing global variables
global predictionPlot 
predictionPlot = 0

# Function to reset the radar device
def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    time.sleep(1)

def main():
    
    device_name = "COM5"
    enable_recording = 1
    output_dir = 'C:/Users/2680850K/Desktop/XeThru Recordings/'

    # Establish connection to XeThru radar
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    print("receive ping, the value is:", hex(xep.ping()))
    fwid = xep.get_system_info(2)

    session_id = 'liam'
    # Display system info
    print('FirmWareID = ', fwid)
    print('Version = ', xep.get_system_info(3))
    print('Build = ', xep.get_system_info(4))
    print('SerialNumber = ', xep.get_system_info(6))
    print('VersionList = ', xep.get_system_info(7))
    fps = 50

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
        RecordingOptions1.set_session_id(session_id)
        baseband_ap_type = DataType.FloatDataType
        recorder.start_recording(baseband_ap_type, output_dir)
    
    xep.x4driver_set_fps(fps)
    infocache = np.empty((0,0))
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    line1, = ax1.plot([], label='Amplitude', linestyle='-')  # Added label, linestyle, and marker
    ax1.set_title('Average Amplitude of IQ Data Over Time')
    ax1.set_ylabel('Average Amplitude')
    ax1.grid(True)
    legend1 = ax1.legend()  # Display legend
    legend1.set_visible(False)  # Initially hide legend

    line2, = ax2.plot([], label='Phase', linestyle='-')  # Added label, linestyle, and marker
    ax2.set_title('Average Phase of IQ Data Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Phase (Degree)')
    ax2.grid(True)
    legend2 = ax2.legend()  # Display legend
    legend2.set_visible(False)  # Initially hide legend
    
    AmpReading = []
    PhaseReading = []
    AmpReadingNew = []
    
    model = load_model(r'C:\Users\2680850K\Desktop\XeThru Recordings\model_8January.h5')
    
    window_size = 91  # Adjust the window size to your preference

    update_interval = 1  # Update the plot every 1 second, adjust as needed
    last_update_time = time.time()

    root = tk.Tk()
    root.title("Human Position Dashboard")
    
    # Load sitting and standing images
    sitting_image = Image.open("sitting.png")
    standing_image = Image.open("standing.png")
    
    # Convert images for displaying in tkinter
    sitting_photo = ImageTk.PhotoImage(sitting_image)
    standing_photo = ImageTk.PhotoImage(standing_image)
    
    # Create label to display the image
    image_label = tk.Label(root)
    image_label.pack()
    
    def update_image(position):
        if position == 'sitting':
            image_label.config(image=sitting_photo)
        elif position == 'standing':
            image_label.config(image=standing_photo)
    
    def update_data():

        while True:
            # For data collection
            contentID = xep.read_message_data_float().content_id
            info = xep.read_message_data_float().info
            data = np.asarray(xep.read_message_data_float().data)
            data_length = len(data)

            i_vec = data[0:int(data_length/2)]
            q_vec = data[int(data_length/2):]
            iq_vec = i_vec + 1j * q_vec
            amplitude = np.abs(iq_vec)
            phase = np.angle(iq_vec)
            phase = np.rad2deg(phase)

            AmpReading.extend(amplitude.ravel())
            PhaseReading.append(np.mean(phase))
            AmpReadingNew.append(np.mean(amplitude))
            print('Length', len(AmpReading))

            if len(AmpReading) >= window_size:  # Display sliding window segments
                input_data = np.array(AmpReading[-window_size:]).reshape(1, -1)
                logits = model.predict(input_data)
                predicted_class = tf.argmax(logits, axis=1).numpy()[0]
                predictionPlot = predicted_class
                ax1.set_title(f'Average Amplitude of IQ Data Over Time => Class Prediction {predictionPlot}')
            
            # Applying the Moving Average filter
            current_time = time.time()

            if current_time - last_update_time > update_interval:
                # Denoise the signal using the Moving Average filter
                filtered_signal = np.convolve(AmpReadingNew, np.ones((window_size,))/window_size, mode='valid')
                    # Update the plot with the denoised signal
                line1.set_ydata(filtered_signal[-200:])
                line1.set_xdata(range(len(filtered_signal[-200:])))

                if predictionPlot == 0:
                    update_image('Empty')
                    line1.set_color('green')
                    legend1 = ax1.legend(['Empty Seat'], loc='upper right')  # Update legend with 'Empty' label
                    legend1.set_visible(True)  # Show legend
                elif predictionPlot == 1:
                    update_image('Standing')
                    line1.set_color('red')
                    legend1 = ax1.legend(['Standing Position'], loc='upper right')  # Update legend with 'Standing' label
                    legend1.set_visible(True)  # Show legend
                else:
                    line1.set_color('blue')
                    update_image('sitting')
                    legend1 = ax1.legend(['Sitting Position'], loc='upper right')  # Update legend with 'Standing' label
                    legend1.set_visible(True)  # Show legend

                ax1.relim()
                ax1.autoscale_view(True, True, True)

                line2.set_ydata(PhaseReading)
                line2.set_xdata(range(len(PhaseReading)))
                ax2.relim()
                ax2.autoscale_view(True, True, True)

                plt.draw()
                plt.show(block=False)
                plt.pause(0.001)

                last_update_time = current_time
            

if __name__ == "__main__":
    main()