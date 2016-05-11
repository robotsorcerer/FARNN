%% Load the various data sets into memory
clear all; clc
cd('/home/local/ANT/ogunmolu/Documents/NNs/FARNN/data')

%The following datasets are publicly available from the Daisy Lab website
% <http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html>
% Process Industry systems
destill = load('destill.dat');
glassfurnace = load('glassfurnace.dat');
powplant = load('powerplant.dat');
evaporator = load('evaporator.dat');
pHdata = load('pHdata.dat');
distill2 = load('distill2.dat');
dryer2 = load('dryer2.dat');
exchanger = load('exchanger.dat');
winding = load('winding.dat');
cstr = load('cstr.dat');
steamgen = load('steamgen.dat');

% Mechanical Systems
ballbeam = load('ballbeam.dat');
hairdryer = load('dryer.dat');
cdplayer = load('CD_player_arm.dat');
flutter = load('flutter.dat');
robotArm = load('robot_arm.dat');
flexibleStructure = load('flexible_structure.dat');

% Biomedical Systems
foetal = load('foetal_ecg.dat');
tongueDisp = load('tongue.dat');


% Environmental Systems
lakeErie = load('erie.dat');

% Thermnic systems
thermic = load('thermic_res_wall.dat');
heating = load('heating_system.dat');

% Time Series Data
timeSeries = load('internet_traffic.dat');

