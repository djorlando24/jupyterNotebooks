%% Rayleigh-Plesset Model with Insoluble Gas

% James Puli, Daniel Duke
% Multiphase Flow Laboratory
% Department of Mechanical & Aerospace Engineering
% Monash University, Australia

% (c) 2026 Monash University

% TODO:
% Set correct averaging window to match our geometry and velocity
% Normalize the histograms to obtain PDFs
% Set correct partial pressure range of insoluble gas

clear all;
close all;
clc;


rng('default');

%% Set Constants
surfTensL = 72e-3; % N/m -- Surface tension of water at 20C
densL = 998; % kg/m^3 -- Density of water at 20C
viscL = 0.89e-3; % m^2 / s -- Kinematic viscocity of water at 20C
PV = 3.189e3; % Pa -- Partial pressure of the vapour phase of the bubble. Read from sat. steam table.
kG = 1.289; % Polytropic index of the insoluble gas

nozzD = 1.7e-3; % m
Pinlet=7.5e5; % Pa inlet stag'n pressure to match Sanjiv's experiment
Pback=1.1e5; % Pa back stag'n pressure to match Sanjiv's experiment
massFlowrate=0.0423; % kg/s measured in Sanjiv's experiment
Ubar = massFlowrate/(densL*0.25*nozzD^2); % bulk velocity estimate
Pnoz=Pinlet - 0.5*densL*(Ubar^2); % Pa static pressure in nozzle at Ubar velocity
if Pnoz < PV
    Pnoz = PV; % floor on static pressure ( no negative pressures please )
end

sample_tmin = (7.9e-3 - 5e-3)/Ubar % 0; % x1/Ubar
sample_tmax = (7.9e-3 + 5e-3)/Ubar %1e-5; % x2/Ubar

%% Parameters to span parametrically

% Experimental  bubble distribution 
% Sanjiv data Fitting parameters: [ mu(µm) sigma scale ]
%K = 1.17; [CO2] =  0 mol/m$^3$: [0.85850511 0.54276889 0.30165182]
%K = 1.17; [CO2] = 25 mol/m$^3$: [0.44638487 0.27163368 0.20521456]
expt_distr_degas = real(random('LogNormal',log(0.85850511e-5), 0.54276889, [1,1000]));
expt_distr_25gas = real(random('LogNormal',log(0.44638487e-5), 0.27163368, [1,1000]));

%Distribution of initial bubble radius upon ejection from nozzle
%R0s = normrnd(7.22e-6,2e-6,[1,1000]); % Gaussian
R0s = expt_distr_25gas;

t0s = -normrnd(5e-5,1e-5,[1,1000]);   % Normal distribution of starting time (i.e. phase shift in bubble oscillation)

%PG0s = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10.000]; % Partial pressure of the insoluble gas in the bubble
PG0s = [0.01, 0.1,1.0,10.0,100.0,1000.0]; % Partial pressure of the insoluble gas in the bubble

% Parametric outer loop (insoluble gas pressures)
for j = 1:length(PG0s)
    bubbles = [];

    % Parametric inner loop (bubble initial radii)
    for i = 1:length(R0s)

        % Initial Conditions
        Pinf = @(t) heaviside(t - 0) * (Pback-Pnoz) + Pnoz; % nozzle and downstream static pressures are set here
        R0 = R0s(i); % m -- Bubble initial size R(0)
        Rd0 = 0; % m/s -- Interface initial velocity dR/dt(0)
        
        PG0_stable = Pinf(-1)-PV+2*surfTensL/R0; % steady state solution from RP equation
        PG0 = PG0_stable*PG0s(j); % Pa -- Initial partial insoluble gas pressure

        % Solve the 1D R-P equations 
        x0 = [R0, 0]; % Integration initial conditions. Velocity and accel. of interface set to 0
        odeFun = @(t, x) rayleighPlesset(t, x, PV,viscL,surfTensL,densL, R0, kG, Pinf, PG0)'; % equations to solve
        opts = odeset('InitialStep', 1e-7, 'MaxStep', 1e-4); % ode time stepping settings
        [t,x] = ode89(odeFun, [t0s(i),sample_tmax], x0, opts); % solve the nonlinear ODE using high order RK
        bubbles = [bubbles, bubble_sample(x,t,sample_tmin,sample_tmax,500)]; % average bubble behaviour over a time window as they pass the measurement zone downstream
    end

    % plotting
    figure(2)
    hold on
    [f,xi] = ksdensity(bubbles,'NumPoints',256);
    % label each plot with the current PG0 iteration factor
    plot(xi,f,'DisplayName', sprintf('PG0 factor = %.3g', PG0s(j)),'LineWidth',2);
    xlabel("Bubble Radius (m)")
    ylabel("Count, Arbitrary")
    
    figure(3)
    hold on
    % label each plot with the current PG0 iteration factor
    h1 = histogram(bubbles,'DisplayName', sprintf('PG0 factor = %.3g', PG0s(j)));
    h1.BinWidth = 2e-7;
    xlabel("Bubble Radius (m)")
    ylabel("Count, Arbitrary")
 
    % exporting data with struct that has unique name
    newName=sprintf('pdf_PG0_%i',j);
    S.(newName) = [xi; f];
    if exist('RPsolution.mat', 'file') == 2
        save('RPsolution.mat','-struct','S','-append');
    else
        save('RPsolution.mat','-struct','S');
    end
end

% show legend and comparative experimental data 
figure(2);
[f,xi] = ksdensity(expt_distr_degas,'NumPoints',256);
plot(xi,f,'DisplayName', 'Expt-Degassed' ,'LineWidth',1);
[f,xi] = ksdensity(expt_distr_25gas,'NumPoints',256);
plot(xi,f,'DisplayName', 'Expt-25 mol/m3' ,'LineWidth',1);
legend('show','Location','best');
xscale log;
yscale log;

figure(3);
h1 = histogram(expt_distr_degas,'DisplayName', 'Expt-Degassed');
h1.BinWidth = 2e-7;
h1 = histogram(expt_distr_25gas,'DisplayName', 'Expt-25 mol/m3');
h1.BinWidth = 2e-7;
legend('show','Location','best');

% show time pressure history of the last case.
Legend = cell(3,1);
figure(1)
subplot(2,1,1)
grid on
yyaxis left
plot(t, x(:,1))
title("Growth of bubble under sinusoidal pressure")
xlabel("Time (s)")
ylabel("Bubble diameter (m)")
yyaxis right
plot(t, Pinf(t))
ylabel("Boundary Pressure (Pa)")
subplot(2,1,2)

save('RPsolution.mat','PG0s','-append');

function dxdt = rayleighPlesset(t,x,PV,viscL,surfTensL,densL, R0, k, Pinf, PG0)
    % t=time
    % x    = [ R     dR/dt ]
    % dxdt = [ dR/dt d^2R/dt^2 ]
    % PV = vapour pressure
    
    dxdt(1) = x(2); % dR/dt
    dxdt(2) = (PV - Pinf(t) + PG0 * (R0 / x(1))^(3*k) - (2*surfTensL + 4 * viscL * x(2))/(x(1)))/(densL * x(1)) - 1.5 * x(2)*x(2)/x(1);
    %dxdt(3) = -(1/x(1))*((3/2)*x(2)^2+(4*viscL*x(2) / x(1)) + (2*surfTensL/(densL*x(1))+P(t,x(1))/densL));
end


function x_interp = bubble_sample(x,t,tmin, tmax, n)
    t_interp = linspace(tmin, tmax, n);
    x_interp = interp1(t, x(:,1), t_interp);
end