% Walking Robot -- DDPG Agent Training Script (2D)
% Copyright 2019 The MathWorks, Inc.

% Simulink.BlockDiagram.getAlgebraicLoops(mdl);
clc; clear;
%% Load simscape params
Assembly_simplified_DataFile

%% SET UP ENVIRONMENT
Ts = 0.025; % Agent sample time
Tf = 10;    % Simulation end time

% Speedup options
useFastRestart = true;
useGPU = true;
useParallel = true;

% Create the observation info
numObs = 6;
observationInfo = rlNumericSpec([numObs 1]);
observationInfo.Name = 'observations';

% create the action info
numAct = 2;
actionInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit', 1);
actionInfo.Name = 'action';

% Environment
mdl = 'TiltRotor_2D_RL';
load_system(mdl);
blk = [mdl,'/RL Agent'];
env = rlSimulinkEnv(mdl,blk,observationInfo,actionInfo);
% env.ResetFcn = @(in)walkerResetFcn(in,upper_leg_length/100,lower_leg_length/100,h/100,'2D');
if ~useFastRestart
   env.UseFastRestart = 'off';
end

%% CREATE NEURAL NETWORKS
createDDPGNetworks;
                     
%% CREATE AND TRAIN AGENT
createDDPGOptions;
agent = rlDDPGAgent(actor,critic,agentOptions);
trainingResults = train(agent,env,trainingOptions)

%% SAVE AGENT
reset(agent); % Clears the experience buffer
curDir = pwd;
saveDir = 'savedAgents';
cd(saveDir)
save(['trainedAgent_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'agent');
save(['trainingResults_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'trainingResults');
cd(curDir)