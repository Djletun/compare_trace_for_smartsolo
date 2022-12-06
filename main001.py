import conv
from matplotlib import pyplot as plt
import struct
import numpy as np

def BCDto(b):
    a = int(b // 16) * 10
    a = a + b % 16
    return a

def read_byte(p, f):
    f.seek(p)
    byte1 = f.read(1)
    return int.from_bytes(byte1, 'little')

def read_SampleRate(f):
    SampleRate = BCDto(read_byte(22, f))
    if SampleRate == 4:
        SampleRate = 0.25
    elif SampleRate == 8:
        SampleRate = 0.5
    elif SampleRate == 10:
        SampleRate = 1
    elif SampleRate == 20:
        SampleRate = 2
    elif SampleRate == 40:
        SampleRate = 4
    return SampleRate


def read_RecordLength(f):
    f.seek(46)
    byte3 = f.read(3)
    RecordLength = int.from_bytes(byte3, 'big')
    return RecordLength

def read_ExtdHPos(f):
    ChannelSet = BCDto(read_byte(28, f))
    ExtdHPos = 32 * 3 + 32 * ChannelSet
    return ExtdHPos

def read_NbofSamplesInTrace(f):
    #NbofSamplesInTrace = read_4byte(read_ExtdHPos(f) + 32, f)
    NbofSamplesInTrace= read_RecordLength(f)//read_SampleRate(f)+1
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

NewFileName='00056979.segd'
NewFileName='00056240.segd'
file_='D:\\kleiman\\бескабельные системы\\SmartSolo\\сейсмические данные\\seismo\\for_AUX\\'+NewFileName
file_='D:\\kleiman\\бескабельные системы\\SmartSolo\\сейсмические данные\\seismo\\segd_428xl\\'+NewFileName


NewFileName_smart_solo='00056979_minus_4ms.segd'
file_smart_solo='D:\\kleiman\\бескабельные системы\\SmartSolo\\сейсмические данные\\seismo\\for_AUX\\'+NewFileName_smart_solo
NewFileName_smart_solo='00056240.00005551.00000001.00056240.segd'
file_smart_solo='D:\\kleiman\\бескабельные системы\\SmartSolo\\сейсмические данные\\seismo\\rev2\\'+NewFileName_smart_solo


with open(file_,'rb') as f:
    SR=read_SampleRate(f)
    trace_ = read_trace(f,0)
    f.close()

permanent_value=np.mean(trace_[2000:3000])
print(permanent_value)

trace_=trace_-permanent_value
K1=2.697*10**(-4)
print('permanent_value', permanent_value*K1)
avr_value=np.mean(trace_[100:150])
print('-avr_value', -avr_value)

print('-avr_value * K1', -1*avr_value*K1)

trace_=-1/avr_value*trace_

with open(file_smart_solo,'rb') as f:
    trace_smart_solo = read_trace(f,0)
    f.close()
avr_value_s=np.mean(trace_smart_solo[100:150])
trace_smart_solo=-1/avr_value_s*trace_smart_solo
print('-avr_value_s',-avr_value_s)

#fig, ax= plt.subplots()
fig = plt.figure(1)
fig = plt.figure(dpi=300)
gridsize = (4, 1)
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=1)
ax1.set_title(NewFileName, fontsize=8)
ax1.grid(True,color='blue',linewidth = 0.2)
ax1.plot(range(len(trace_)), trace_, c='black', linewidth=0.5)
#ax1.plot(range(400,550), trace_[400:550], c='black', linewidth=0.5)

ax2 = plt.subplot2grid(gridsize, (1, 0))
ax2.set_title(NewFileName_smart_solo, fontsize=8)
ax2.grid(True,color='blue',linewidth = 0.2)
ax2.plot(range(len(trace_smart_solo)), trace_smart_solo, c='blue', linewidth=0.5)
#ax2.plot(range(400,550), trace_smart_solo[400:550], c='blue', linewidth=0.5)

ax3 = plt.subplot2grid(gridsize, (2, 0))
ax3.set_title('two signals', fontsize=8)
ax3.grid(True,color='blue',linewidth = 0.2)
#sub_=trace_[400:550]-trace_smart_solo[400:550]
#sub_=trace_-trace_smart_solo
#ax3.plot(range(len(sub_)), sub_, c='green', linewidth=0.5)
#ax3.plot(range(400,550), sub_, c='green', linewidth=0.5)

delta_time=2


ax3.plot(range(0,550), trace_[0:550], c='black', linewidth=0.5)
ax3.plot(range(0,550), trace_smart_solo[delta_time:550+delta_time], c='blue', linewidth=0.5)

ax4 = plt.subplot2grid(gridsize, (3, 0))
ax4.set_title('difference', fontsize=8)
ax4.grid(True,color='blue',linewidth = 0.2)
sub_=trace_[0:550]-trace_smart_solo[0+delta_time:550+delta_time]
#sub_=trace_-trace_smart_solo
#ax4.plot(range(400,550), sub_[400:550], c='green', linewidth=0.5)
ax4.plot(range(0,550), sub_, c='green', linewidth=0.5)



spectr_plot = plt.figure(2)
spectr_plot = plt.figure(dpi=300)
ax5=plt.subplot()
ax5.grid(True,color='blue',linewidth = 0.2)
ax5.set_title(NewFileName, fontsize=8)

spectr_sercel = np.fft.rfft(trace_[0:550])
ax5_x=np.fft.rfftfreq(len(trace_[0:550]),SR/1000)
ax5_y=20*np.log10(np.abs(spectr_sercel)/max(np.abs(spectr_sercel)))
ax5.plot(ax5_x,ax5_y,c='black', linewidth=0.5)

spectr_smartsolo = np.fft.rfft(trace_smart_solo[delta_time:550+delta_time])
ax5_x=np.fft.rfftfreq(len(trace_smart_solo[delta_time:550+delta_time]),SR/1000)
ax5_y=20*np.log10(np.abs(spectr_smartsolo)/max(np.abs(spectr_smartsolo)))
ax5.plot(ax5_x,ax5_y,c='blue', linewidth=0.5)



#plt.savefig(NewFileName + '.pdf', dpi=300)
plt.show()