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




def average(lst):
    return sum(lst) / len(lst)



def signal_update(iq_vec, infocache, channel, threshold, fs_sample,pca):
    

    threshold_bin = fs_sample * threshold
    if infocache.shape[1] > threshold_bin:
        analysis_signal = np.concatenate((iq_vec,infocache[:,0:(threshold_bin-1)]),axis=1)
        #print ('analysis_signal',analysis_signal.shape)
        infocache = analysis_signal[0:threshold_bin-25]

        analysis_signal_heart = np.abs(analysis_signal)
        analysis_signal_resp = np.angle(analysis_signal)

        pca_coef_heart = pca.fit_transform(analysis_signal_heart[0:channel,:])
        pca_coef_resp = pca.fit_transform(analysis_signal_resp[0:channel,:])

        analysis_signal_heart_sum = np.transpose(analysis_signal_heart)
        analysis_signal_resp_sum = np.transpose(analysis_signal_resp)

        pca_coef_heart_sum = pca_coef_heart[:,0:1]
        pca_coef_resp_sum = pca_coef_resp[:,0:1]

        output_resp = np.matmul(analysis_signal_resp_sum, pca_coef_resp_sum)
        output_heart = np.matmul(analysis_signal_heart_sum, pca_coef_heart_sum)

        

    else:
        if infocache.shape[0] < channel:
            infocache = iq_vec
            output_resp = np.empty((0,0))
            output_heart = np.empty((0,0))
        else:
            infocache = np.concatenate((iq_vec,infocache),axis=1)
            output_resp = np.empty((0,0))
            output_heart = np.empty((0,0))
            

    return output_resp,output_heart,infocache



def heart_detect(output_heart,FPS):
    ############### Heart Rate #####################################
    u_heart ,  u_hat_heart ,  omega_heart  = VMD(output_heart,K=9,alpha = 20,tau = 0,DC = 0,init = 0,tol = 1e-6)
    # u is correct as u_hat same shape but is complex values
    u_heart = np.transpose(u_heart)
    vmd_signal_heart = np.sum(u_heart, axis=1)
    #Apply Filter to the data
    sos = butter(10, [1,1.8], btype="bandpass", output="sos",fs=FPS)
    heart_sig = sosfiltfilt(sos, vmd_signal_heart)
    xc = heart_sig - heart_sig.mean()
    n = len(xc) #samples
    dF = FPS/n # hertz per sample
    f = np.array(range(0,FPS*10+1,int(dF*10)))/10
    f = f[0:(int(len(f)/2))]
    Y1_fft = np.fft.fftshift(np.fft.fft(xc)) / n
    number = len(Y1_fft)/2+1
    liamfloor = math.floor(number)
    fft_abs = Y1_fft[liamfloor:]
    #Peak detection
    lv = np.array(range(1,len(f)))
    min_dist = math.ceil(20*dF)
    fft_abs = np.abs(fft_abs)
    #Distance not height Distance not height Distance not height Distance not height Distance not height Distance not height Distance not height 
    peaks, _ = find_peaks(fft_abs,distance=min_dist)

    readings = []
    for index in peaks :
        readings.append(fft_abs[index])
    
    heart_bpm = np.argmax(readings)
    heart_bpm = (f[peaks[heart_bpm]]*60)

    return heart_bpm


            
def resp_detect(output_resp,FPS):
    ############### Resp Rate ####################################
    fc =  0.5
    wn=2*fc/FPS

    
    output_resp = output_resp.reshape(output_resp.shape[0])
    b, a = butter(2, wn,btype='lowpass')
    output_resp = filtfilt(b, a, output_resp,padlen=len(output_resp)-1)
    
    output_resp = output_resp - output_resp.mean()

    # Estimate without filter bias
    # number1=(0.1*len(output_resp))
    # liamfloor1 = math.floor(number1)
    # output_resp = output_resp[liamfloor1:]
    # respcorr = np.correlate(output_resp,output_resp,'full')
    # min_dist1 = math.ceil(0.6*FPS)
    # peaks1, _ = find_peaks(respcorr,height=min_dist1)

    min_dist1 = 0.1*(max(output_resp)-min(output_resp))
    peaks1, _ = find_peaks(output_resp,height=min_dist1)

    if len(peaks1) > 1:
        peaksgradient = np.gradient(peaks1)
        
        T=1/FPS
        avg_dt2 = np.mean(peaksgradient)*T
        avg_pd = 1/avg_dt2


        resp_rate = avg_pd*60
    else:
        resp_rate = 0

    return resp_rate



def main():
    device_name = "COM5"
    enable_recording = 1
    output_dir = 'C:/Users/2680850K/Desktop/XeThru Recordings/'
    

    heart_readings = []
    lung_readings = []
    avg_heart_readings = []
    avg_lung_readings = []    

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
    fps = 25

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
    ax1.set_title('Average Amplitude of IQ Data Over Time')
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
        avg_amplitude = np.mean(amplitude, axis=0)  # Averaging over the second dimension
        AmpReading.append(avg_amplitude)  # Store the average amplitude
        print('Amplitude length ',len(AmpReading))

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


        
        #Do the average
        # if output_resp.shape[0] >= 1 and output_heart.shape[0] >= 1:
        #     time.sleep(0.05)
        #     resp_rate = resp_detect(output_resp,fps)
        #     heart_rate = heart_detect(output_heart,fps)
        #     print ('resp_rate', resp_rate)
        #     print ('heart_rate', heart_rate)

            

        #     heart_readings.append(heart_rate)
        #     heart_readings = heart_readings[-100:]
        #     avg_heart_bpm = average(heart_readings)
        #     avg_heart_readings.append(avg_heart_bpm)
        #     avg_heart_readings = avg_heart_readings[-100:]
            


        #     lung_readings.append(resp_rate)
        #     lung_readings = lung_readings[-100:]
        #     avg_breathing_bpm = average(lung_readings)
        #     avg_lung_readings.append(avg_breathing_bpm)
        #     avg_lung_readings = avg_lung_readings[-100:]

            # Plot amplitude
            # def makeFig():
            #     plt.figure(figsize=(10, 4))
            #     plt.title('Amplitude of IQ Data')
            #     plt.plot(AmpReading)
            #     plt.xlabel('Time')
            #     plt.ylabel('Amplitude')
            #     plt.grid(True)
            # drawnow(makeFig)
            # def makeFig():
                

            #     # plt.subplot(211)
            #     # plt.title(f'Average Heart rate readings - {round(avg_heart_bpm,2)} BPM')
            #     # plt.ylim(0,150) 
            #     # plt.plot(avg_heart_readings,'r',label='Heart rate')
            #     # plt.grid(True)
            #     # plt.legend(loc='lower left')
            
            #     plt.subplot(212)
            #     plt.tight_layout(pad=2.0)
            #     plt.title(f'Average Breathing rate readings - {round(avg_breathing_bpm,2)} BPM')
            #     plt.ylim(0,50) 
            #     plt.plot(avg_lung_readings,'b',label='Breathing rate')
            #     plt.grid(True) 
            #     plt.legend(loc='lower left')

            # drawnow(makeFig)



            

        x = x + 1
    





if __name__ == "__main__":
    main()


        
