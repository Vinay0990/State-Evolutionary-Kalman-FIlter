clear all;
close all;
%% KALMAN ESTIMATOR
% PART OF EE525 COURSE JAN-APRIL 2015
% Instructor/coder: Dr. Kannan Karthik

%% The number of iterations 
L = 200;
r = 1; % If the trajectory is circular r is ONE, else r is within the range [0,1] for a SPIRAL
p = randn(1,1) > 0;
theta0 = 0*pi;
% Resolution in THETA; k = 26 = 24.28, 51 =24.75, 99 = 24.41
k = 26;
theta = 2*k*pi/200;
A = r*[cos(theta) sin(theta); -sin(theta) cos(theta)];
X = [cos(theta0); sin(theta0)];
Y = X;
SIG_W = 0.2; % STATE NOISE VARIANCE
SIG_V = 1; % OBSERVATION NOISE VARIANCE

for k = 1:1000, % NUMBER OF ITERATIONS to FIND MEAN CONVERGENCE TIME
    
% STATE EVOLUTION and OBSERVED VECTORS    
for i = 2:L,
    W = SIG_W*[randn(1,1);randn(1,1)]; % REALIZATION OF THE STATE NOISE PROCESS
    V = SIG_V*[randn(1,1);randn(1,1)]; % REALIZATION OF THE OBSERVATION NOISE PROCESS
    X(:,i) = A*X(:,i-1) + W; % NATURAL STATE EVOLUTION
    Y(:,i) = X(:,i) + V ; % OBSERVATIONS
end
M_SIG_V = [SIG_V^2 0; 0 SIG_V^2]; % ACTUAL NOISE COVARIANCE MATRIX
M_SIG_W = [SIG_W^2 0; 0 SIG_W^2]; % ACTUAL STATE NOISE COVARIANCE MATRIX
X_hat = [-1;0]; % INITIALIZATION OF THE KALMAN ESTIMATE X_hat_0/0
X_pred_hat = [-1;0]; % ONE STEP AHEAD PREDICTION INITIALIZATION
E(1) = 1;
M_STATE_old = eye(2); % This corresonds to the error covariance matrix SIGMA_K-1/K-1 set at K-1 = 0;
a = 1; % Some form of initialization;
VAR_e0 = trace(A*M_STATE_old*A' + M_SIG_W); % This correponds to the initial value of SIGMA_K/K-1
G(1) = VAR_e0/(VAR_e0 + a); % KALMAN GAIN INITIALIZATION
MV_cum = eye(2); % Initilization of the NOISE VARIANCE MATRIX

for i = 2:L, % SERIAL ARRIVAL OF OBSERVATION VECTORS
    OBSp = Y(:,i); % CURRENT OBSERVATION VECTOR
    OBSp_perp = OBSp - A*X_hat(:,i-1); % INNOVATION DERIVED FROM THE CURRENT OBSERVATION VECTOR
    M_OBS = OBSp_perp*OBSp_perp'; % CURRENT INNOVATION COVARIANCE MATRIX
    X_hat(:,i) = A*X_hat(:,i-1) + G(i-1)*OBSp_perp; % MAIN KALMAN ESTIMATOR EQUATION
    X_pred_hat(:,i) = A*X_hat(:,i); % ONE STEP AHEAD PREDICTION EQUATION
    M_STATE_new = (1-2*G(i-1))*A*M_STATE_old*A' + M_SIG_W + G(i-1)^2*M_OBS; % ERROR COVARIANCE UPDATION EQUATION
    MVk = M_OBS - A*M_STATE_old*A' - M_SIG_W; % NOISE COVARIANCE ESTIMATE FROM CURRENT INNOVATION
    MV_cum = ((i-1)/i)*MV_cum + (1/i)*MVk; % CUMULATIVE NOISE COVARIANCE ESTIMATE FROM ALL PAST INNOVATIONS
    a = trace(MV_cum); % THE SCALAR EFFECTIVE NOISE VARIANCE
    G(i) = trace(M_STATE_new)/(trace(M_STATE_new) + a); % KALMAN GAIN UPDATE
    E(i) = trace(M_STATE_old); % QUANTIFIED ERROR DERIVED FROM THE COVARIANCE MATRIX
    M_STATE_old = M_STATE_new;% TRANSFER OF VARIABLE from SIGMA_k-1/k-1 to SIGMA_k/k
end

% TERMINATION POINT to detect convergence
for i = 1:L-1,
    R = abs((E(i) - E(i+1))/E(i))*100;
    if R < 1,
        %disp('The CONVERGENCE TIME IN UNITS IS...');
        i;
        break;
    end
end
T(k) = i;
end
disp('Mean convergence time');
mean(T)

% Case when STATE EVOLUTION IS A CIRCULAR MOVEMENT; PROBLEM IS ANALOGOUS TO
% TRACKING PHASE
figure;
title('PHASE TRACKING FOR CIRCULAR MOVEMENTS');
plot(atan(X(2,:)./X(1,:)), 'b');  hold on;
plot(atan(Y(2,:)./Y(1,:)), 'r'); hold on;
plot(atan(X_pred_hat(2,:)./X_pred_hat(1,:)), 'g'); hold on;

% ACTUAL AND ESTIMATED STATE TRAJECTORIES 
figure;
B = X;
line(B(1,:),B(2,:),'Marker','*','LineStyle','-'); hold on;
xlabel([r,theta]);
grid on;
title('Original State Evolution');

figure;
B = Y;
line(B(1,:),B(2,:),'Marker','*','LineStyle','-'); hold on;
grid on;
title('Observed state evolution');

figure;
B = X_hat;
line(B(1,:),B(2,:),'Marker','*','LineStyle','-'); hold on;
grid on;
title('Estimated state evolution');

figure; stem(E);
grid on;
title('State tracking error');


%% OVERLAID PLOTS OF ACTUAL AND ESTIMATED STATE TRAJECTORIES
figure;
B = X;
line(B(1,:),B(2,:),'Marker','>','LineStyle','-'); hold on;
xlabel([r,theta]);
grid on;
title('Original State Evolution');

B = X_pred_hat;
line(B(1,:),B(2,:),'Marker','*','LineStyle','-'); hold on;
grid on;
title('Estimated state evolution');
