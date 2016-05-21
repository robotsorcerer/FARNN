%% Load Torch Trained Network
clear all; clc; close all
cd('/home/lex/Documents/FARNN/network');
% load lstm trainer
lstmID = fopen('lstm-net.t7');
lstm = fread(lstmID);
fclose(lstmID);
whos lstm


%load simple rnn
rnnID = fopen('rnn-net.t7');
rnn = fread(rnnID);
fclose(rnnID);
whos rnn

% load mlp trainer
mlpID = fopen('rnn-net.t7');
mlp = fread(mlpID);
fclose(mlpID);
whos mlp
