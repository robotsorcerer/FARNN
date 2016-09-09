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
% Data from a flexible robot arm; the arm is installed on an electrical
% motor. Reaction torque of the structure on the ground to acceleration o
% fthe flexible arm input is a periodic sine wave
clear all; clc
username = system('whoami');
cd('/home/lex/Documents/NNs/FARNN/data')
softRobot = load('softRobot.mat')
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

%softRobot
SR = softRobot.pose;
srInput = SR(:,1);
srOut = SR(:,2:end);

%% Train ballbeam
    clc
    x = num2cell(ballbeam(:,1));
    t = num2cell(ballbeam(:,2));
 setdemorandstream(491218381);
 
 net = narxnet(1:2,1:2,10);
 view(net)
 [Xs,Xi,Ai,Ts] = preparets(net,x,t,{});
 [net,tr] = train(net,Xs,Ts,Xi,Ai);
 nntraintool
 plotperform(tr)

%% estiomate robotArm params

                               
 % Import   roboArm                                                           
 roboArmd = detrend(robotArm,0)                                                
 roboArmdd = detrend(roboArmd,1)                                              
                                                                              
 Opt = arxOptions;                                                            
 Opt.InitialCondition = 'estimate';                                           
                                                                              
 [L, R] = arxRegul(roboArmdd, [10 10 1], arxRegulOptions('RegulKernel','SE'));
 Opt.Regularization.Lambda = L;                                               
 Opt.Regularization.R = R;                                                    
 arx10101 = arx(roboArmdd,[10 10 1], Opt, 'IntegrateNoise', true); 
 % Train ballbeam
 
 %% Maglev
 clear all; clc
 [x, t] = maglev_dataset;
 setdemorandstream(491218381);
 
 net = narxnet(1:2,1:2,10);
 view(net)
 [Xs,Xi,Ai,Ts] = preparets(net,x,{},t);
 [net,tr] = train(net,Xs,Ts,Xi,Ai);
 nntraintool
 plotperform(tr)
 %% Neural Network saved models
 % Soft Robot Data
 clc; clear all;
 cd('/home/lex/Documents/NNs/FARNN/data');
 
 mlp_srID = fopen('softRobot_mlp-net.t7');
 mlp_sr = fread(mlp_srID);
 fclose(mlp_srID);
 %whos mlp_srID

 lstm_srID = fopen('softRobot_lstm-net.t7');
 lstm_sr = fread(lstm_srID);
 fclose(lstm_srID);