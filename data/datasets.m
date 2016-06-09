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

%% load mat files
% Data from a flexible robot arm; the arm ios installed on an electrical
% motor. Reaction torque of the structure on the ground to acceleration o
% fthe flexible arm input is a periodic sine wave
clear all; clc
cd('/home/local/ANT/ogunmolu/Documents/NNs/FARNN/data')
robotArm = load('robotArm.mat');
robotArm = robotArm.robotArm;
robotArm_u = robotArm(:,1);
robotArm_y = robotArm(:,2);

% ballbeam data
ballbeam = load('ballbeam.mat');
ballbeam = ballbeam.ballbeam;

% glassfurnace data
glassfurnace = load('glassfurnace.mat');
glassfurnace = glassfurnace.glassfurnace;
time_g = glassfurnace(:, 1);
input = glassfurnace(:,2:4);
output = glassfurnace(:,5:10);

off = ceil(0.6* length(glassfurnace));
train_input = input(1:off,:);
train_out = output(1:off,:);

train_input(749,:,1)

