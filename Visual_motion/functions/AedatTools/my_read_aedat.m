clear all
addpath('/home/eden/Desktop/AedatTools/Matlab') 

aedat = struct;
aedat.importParams.filePath = '/home/eden/Desktop/AedatTools/data/user18_fluorescent_led.aedat';
aedat = ImportAedat(aedat);

TD.p = aedat.data.polarity.polarity;
TD.x = aedat.data.polarity.x;
TD.y = aedat.data.polarity.y';
TD.ts = aedat.data.polarity.timeStamp;