%% Load Soft Robot Torch Trained Network
clear all; clc; close all
cd('/home/lex/Documents/FARNN/network');
% load lstm trainer
lstmID = fopen('lstm-net.t7');
lstm = fread(lstmID);
fclose(lstmID);
whos lstm

% load fast-lstm trainer
fastlstmID = fopen('fastlstm-net.t7');
fastlstm = fread(fastlstmID);
fclose(fastlstmID);
whos fastlstm

%load simple rnn
rnnID = fopen('rnn-net.t7');
rnn = fread(rnnID);
fclose(rnnID);
whos rnn

% load mlp trainer
mlpID = fopen('mlp-net.t7');
mlp = fread(mlpID);
fclose(mlpID);
whos mlp

%% Load Robot Arm Network
robotArmlstmID = fopen('robotArm_lstm-net.t7');
robotArmlstm = fread(robotArmlstmID);
fclose(robotArmlstmID);
whos robotArmlstm

% load fast-lstm trainer
robotArm_fastlstmID = fopen('robotArm_fastlstm-net.t7');
robotArm_fastlstm = fread(robotArm_fastlstmID);
fclose(robotArm_fastlstmID);
whos robotArm_fastlstm

%load simple rnn
robotArm_gruID = fopen('robotArm_gru-net.t7');
robotArm_gru = fread(robotArm_gruID);
fclose(robotArm_gruID);
whos robotArm_gru

% load mlp trainer
% mlpID = fopen('mlp-net.t7');
% mlp = fread(mlpID);
% fclose(mlpID);
% whos mlp

%% Load ballbeam network
ballbeamlstmID = fopen('ballbeam_lstm-net.t7');
ballbeamlstm = fread(ballbeamlstmID);
fclose(ballbeamlstmID);
whos ballbeamlstm

% load fast-lstm trainer
ballbeam_fastlstmID = fopen('ballbeam_fastlstm-net.t7');
ballbeam_fastlstm = fread(ballbeam_fastlstmID);
fclose(ballbeam_fastlstmID);
whos ballbeam_fastlstm

%load simple rnn
ballbeam_gruID = fopen('ballbeam_gru-net.t7');
ballbeam_gru = fread(ballbeam_gruID);
fclose(ballbeam_gruID);
whos ballbeam_gru

%% Load glassfurnace network
glassfurnacelstmID = fopen('glassfurnace_lstm-net.t7');
glassfurnacelstm = fread(glassfurnacelstmID);
fclose(glassfurnacelstmID);
whos glassfurnacelstm

% load fast-lstm trainer
glassfurnace_fastlstmID = fopen('glassfurnace_fastlstm-net.t7');
glassfurnace_fastlstm = fread(glassfurnace_fastlstmID);
fclose(glassfurnace_fastlstmID);
whos glassfurnace_fastlstm

%load simple rnn
glassfurnace_gruID = fopen('glassfurnace_gru-net.t7');
glassfurnace_gru = fread(glassfurnace_gruID);
fclose(glassfurnace_gruID);
whos glassfurnace_gru