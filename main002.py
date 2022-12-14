import struct
import os
import numpy as np
from matplotlib import pyplot as plt


def BCDto(b):
    a = int(b // 16) * 10
    a = a + b % 16
    return a


def read_byte(p, f):
    f.seek(p)
    byte1 = f.read(1)
    return int.from_bytes(byte1, 'little')


def read_3byte(p, f):
    f.seek(p)
    byte3 = f.read(3)
    return int.from_bytes(byte3, 'big')


def read_RecordLength(f):
    RecordLength = read_3byte(46, f)
    return RecordLength


def read_ExtdHPos(f):
    ChannelSet = BCDto(read_byte(28, f))
    ExtdHPos = 32 * 3 + 32 * ChannelSet
    return ExtdHPos


def read_NbofSamplesInTrace(f):
    ExndHLength = 32 * BCDto(read_byte(30, f))
    k1 = read_byte(31, f)
    if k1 != 0xff:
        ExnlHLength = 32 * BCDto(k1)
    else:
        k1 = read_byte(39, f)
        k2 = read_byte(40, f)
        ExnlHLength = int(k2) + int(k1) * 256
        ExnlHLength *= 32
    posTraceH = read_ExtdHPos(f) + ExndHLength + ExnlHLength
    NbofSamplesInTrace = read_3byte(posTraceH + 20 + 7, f)
    return NbofSamplesInTrace


def posTraceH(f, TraceNb):
    ExndHLength = 32 * BCDto(read_byte(30, f))
    k1 = read_byte(31, f)
    if k1 != 0xff:
        ExnlHLength = 32 * BCDto(k1)
    else:
        k1 = read_byte(39, f)
        k2 = read_byte(40, f)
        ExnlHLength = int(k2) + int(k1) * 256
        ExnlHLength *= 32
    posTraceH = read_ExtdHPos(f) + ExndHLength + ExnlHLength
    pos = posTraceH + 244 + (244 + read_NbofSamplesInTrace(f) * 4) * TraceNb
    return pos


def read_trace(f, TraceNb):
    k = int(read_NbofSamplesInTrace(f))
    trace = np.empty(shape=[k], dtype=np.float32)
    pos = posTraceH(f, TraceNb)
    f.seek(pos)
    trace_byte = np.empty(shape=[4 * k], dtype=np.uint8)
    trace_byte = f.read(4 * k)
    for i in range(k):
        trace[i] = struct.unpack('>f', trace_byte[4 * i:i * 4 + 4])[0]
    return trace


def spectr_num(data, fs, fft_size, noverlap):
    # data = a numpy array containing the signal to be processed
    # fs = a scalar which is the sampling frequency of the data

    hop_size = fft_size - noverlap
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)
    dt = hop_size / fs

    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size
    # proc=np.array([])
    # proc=np.append(proc,np.zeros(fft_size//2))
    proc = np.zeros(fft_size + len(data))
    proc[fft_size // 2:len(data) + fft_size // 2] = data / fft_size
    # proc=np.append(proc,data/fft_size)
    # proc = np.concatenate((proc, np.zeros(pad_end_size)))

    result = np.empty((fft_size, total_segments), dtype=np.csingle)  # space to hold the result

    for i in range(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples
        result[:fft_size, i] = spectrum[:fft_size]  # append to the results array

    t = np.arange(0, t_max, dt)
    freq = np.arange(0, fs / 2, fs / 2 / fft_size)
    return freq, t, result


############################ получение символа разрыва от версии ОС
if os.name == 'nt':
    symbol = '\\'
if os.name == 'posix':
    symbol = '/'

data_path = os.path.dirname(os.path.abspath(__file__)) + symbol + 'data'
print(data_path)
dir_files = os.listdir(data_path)  # всё содержимое папки dir_path
segd_files = tuple(filter(lambda s: s[s.rfind('.'):] == '.segd', dir_files))  # только файлы segd
if len(segd_files) == 0:
    print('no segd files in dir')
    exit()
print('segd_files ', segd_files)
# создаем папку result
if not os.path.isdir(data_path + 'result'):
    os.mkdir(data_path + 'result')
# 451012744 - 0,3,9
# 451014839 - 13,14,1
# 451013291 - 4,2,5
# 451011140 - 7,6,10
# 451012540 - 8,12,11
file_ = data_path + symbol+segd_files[9]
with open(file_, 'rb') as f:
    trace_as_np_array0 = read_trace(f, 0)
    f.close()
file_ = data_path + symbol+segd_files[1]
with open(file_, 'rb') as f:
    trace_as_np_array1 = read_trace(f, 0)
    f.close()
file_ = data_path + symbol+segd_files[5]
with open(file_, 'rb') as f:
    trace_as_np_array2 = read_trace(f, 0)
    f.close()
file_ = data_path + symbol+segd_files[10]
with open(file_, 'rb') as f:
    trace_as_np_array3 = read_trace(f, 0)
    f.close()
file_ = data_path + symbol+segd_files[11]
with open(file_, 'rb') as f:
    trace_as_np_array4 = read_trace(f, 0)
    f.close()


trace_as_np_array=trace_as_np_array0+trace_as_np_array1+trace_as_np_array2+trace_as_np_array3+trace_as_np_array4

print(len(trace_as_np_array))

fig = plt.figure(0, dpi=300)

ax1 = plt.subplot()

freq, time, Zxx = spectr_num(trace_as_np_array, fs=500, fft_size=128*2, noverlap=64*3+32)
cmap='nipy_spectral'
cmap='magma'
fs=500
autopower = np.abs(Zxx * np.conj(Zxx))
k = np.amax(autopower)
spgraf = 20 * np.log10(autopower / k)
im = ax1.pcolormesh(time, freq, spgraf, cmap=cmap)
# Pxx, freqs, bins, im = ax8.specgram(force/NFFT/2, NFFT=NFFT, Fs=500, noverlap=64*3+32,detrend='mean',mode='psd',scale='dB',cmap=cmap)
fig.colorbar(im).set_label('Intensity, dB')

fig1 = plt.figure(1, dpi=300)

ax1 = plt.plot(range(len(trace_as_np_array)),trace_as_np_array,color='black',linewidth = 0.2)
plt.show()

# trace_as_np_array - массив данных типа np.array
