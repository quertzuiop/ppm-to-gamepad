import pyaudio
import numpy as np
import vgamepad as vg
import cv2

# SETTINGS
DEBUG = False
FORMAT = pyaudio.paInt16        # audio format (bytes per sample), not important
RATE = 200000                   # samples per second, accuracy of input
CHANNELS = 8                    # number of channels
T=20/1000                       # length of each update in s
CHUNK = int(RATE*T*2.5)           # input sample size. When too low, might not detect cycles correctly. Higher values decrease update rate
RESET_PULSE_TRESHOLD = 600      # make this smaller than the length of the reset pulse
channel_mapping = [2, 1, 0, 3]  #channels to joystick axes: [Lx, Ly, Rx, Ry]
#output settings
gamepad = vg.VX360Gamepad()     #VX360Gamepad (xbox 360) or VDS4Gamepad (dualshock4)
JOYSTICK_MAX = 25000            #maximum value of joysticks. 32767 for VX360Gamepad, 255 for VDS4Gamepad

def update_gamepad(channels):
    gamepad.right_joystick(x_value=channels[0], y_value=channels[1])
    gamepad.left_joystick(x_value=channels[2], y_value=channels[3])
    gamepad.update()

def get_data(stream):
    return np.frombuffer(stream.read(CHUNK*2, False), dtype=np.int16) > 1000

def max_streak(data):
    max_streak = 0
    streak = 0
    for i in range(len(data)):
        if data[i]==1:
            streak += 1
        else:
            streak = 0
        if streak > max_streak:
            max_streak = streak
    return max_streak

def calibrate(channels, stream, calibration, channel_mapping, target_scale=1):
    for i, channel in enumerate(["Lx", "Ly", "Rx", "Ry"]):
        input(f"Move the {channel} stick to its maximum position and press enter")
        vals = np.zeros((10, channels))
        for j in range(5):
            vals[j] = ppm_to_channels(channels, get_data(stream))[channel_mapping[i]]
        #print(maximum, i, channel_mapping[i], ppm_to_channels(channels, get_data(stream)))
        input(f"Move the {channel} stick to its minimum position and press enter")
        for j in range(5):
            vals[j+5] = ppm_to_channels(channels, get_data(stream))[channel_mapping[i]]
        maximum = np.max(vals)+1
        minimum = np.min(vals)-1
        print(minimum, maximum, i, channel_mapping[i], ppm_to_channels(channels, get_data(stream)))
        calibration[i][0] = (maximum+minimum)/2 #offset
        calibration[i][1] = (2*target_scale)/(maximum-minimum)#scale
    return calibration

def segment_data(data):
    change_indices = np.where(np.diff(data) != 0)[0] + 1
    return np.diff(np.concatenate(([0], change_indices, [len(data)])))[not data[0]::2]

def ppm_to_channels(channels, data):
    c = np.zeros(channels)
    segments = segment_data(data)
    #print(segments)
    i=0
    while segments[i] < RESET_PULSE_TRESHOLD:
        i+=1
    i+=1
    for channel in range(channels):
        c[channel] = segments[i+channel]
    return c

def set_channels(raw_channels, channel_mapping, calibration):
    c = np.zeros(4, dtype=np.int16)
    for i in range(4):
        val = int((raw_channels[channel_mapping[i]] - calibration[i][0]) * calibration[i][1])
        if np.abs(val) > JOYSTICK_MAX:
            print(f"Warning: channel {i} (raw:{channel_mapping[i]}) value {val} (raw: {raw_channels[channel_mapping[i]]}) exceeds maximum value {JOYSTICK_MAX}. Clipping to maximum value. Calibration {calibration[i]}")
        c[i] = int((raw_channels[channel_mapping[i]] - calibration[i][0]) * calibration[i][1])
    return c

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)
calibration = np.ones((4, 2), dtype=np.float64)
calibration = calibrate(CHANNELS, stream, calibration, channel_mapping, JOYSTICK_MAX)
print(calibration)

while True:
    data_np = get_data(stream)
    raw_channels = ppm_to_channels(CHANNELS, data_np)
    channels = set_channels(raw_channels, channel_mapping, calibration)
    update_gamepad(channels)
    print(channels)
    if DEBUG:
        img = np.full((500, 500, 3), 255, dtype=np.uint8)
        img = cv2.circle(img, (channels[0]//500+100, channels[1]//500+100), 10, (0, 0, 255), -1)
        img = cv2.circle(img, (channels[2]//500+100, channels[3]//500+100), 10, (0, 255, 0), -1)
        cv2.imshow('joystick pos', img)
        cv2.waitKey(1)
        data_np = data_np.astype(np.uint8) * 255
        cv2.imshow('signal', np.tile(data_np[::5], (100, 1)))
        cv2.waitKey(1)
    #print(f"{ppm_to_channels(8, data_np)}\r ")
    #line.set_ydata(data_np)
        