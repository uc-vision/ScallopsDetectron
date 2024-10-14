import crcmod
import math
import numpy as np

C = lambda ang: math.cos(math.radians(ang))
S = lambda ang: math.sin(math.radians(ang))

def YPRToRot33(y, p, r):
    return np.array([[C(y)*C(p), C(y)*S(p)*S(r)-S(y)*C(r), C(y)*S(p)*C(r)+S(y)*S(r)],
                     [S(y)*C(p), S(y)*S(p)*S(r)+C(y)*C(r), S(y)*S(p)*C(r)-C(y)*S(r)],
                     [-S(p), C(p)*S(r), C(p)*C(r)]])

crc = crcmod.predefined.mkPredefinedCrcFun("crc-8")
def check_crc(line):
    data, checksum = line.split(b'*')
    return crc(bytes(data)) == int(checksum, 16)

def parse_line(line):
    if not check_crc(line):
        print('Error Bad CRC')
        return None
    #import ipdb;ipdb.set_trace()
    #sys.exit(0)
    data=line.split(b'*')[0].split(b',')
    ret=None
    if data[0]==b'wrz':
        # Velocity report
        ret={'type':'vel'}
        keys='vx,vy,vz,valid,alt,fom,cov,tov,tot,time,status'.split(',')
        for i in range(len(keys)):
            if i == 3:
                ret[keys[i]]=data[i+1]
            elif i == 6:
                ret[keys[i]]=[float(v) for v in data[i+1].split(b';')]
            else:
                ret[keys[i]]=float(data[i+1])

    if data[0]==b'wru':
        # Transducer report
        ret={'type':'transducer2'}
        keys='id,velocity,distance,rssi,nsd'.split(',')
        for i in range(len(keys)):
            ret[keys[i]]=float(data[i+1])

    if data[0]==b'wrt':
        #Transducer report
        ret={'type':'transducer'}
        ret['dist']=[float(data[i+1]) for i in range(4)]

    if data[0]==b'wrp':
        # Deadreckoning report
        keys='time,x,y,z,pos_std,roll,pitch,yaw,status'.split(',')
        ret={'type':'deadreacon'}
        for i in range(len(keys)):
            #print(keys[i],data[i+1])
            ret[keys[i]]=float(data[i+1]) if i<=8 else data[i+1]

    if data[0]==b'wrn' or data[0]==b'wra':
        # DR reset reply
        ret={'type':'Reset DR reply'}
        ret['res']=str(data[0])

    return ret