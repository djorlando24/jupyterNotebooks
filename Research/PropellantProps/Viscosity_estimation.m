% Weight and Buoyancy force calculation

clc;
clear;

% Constants and ball bearing values
g = 9.81;         % gravitational acceleration
rho_b = 7850;     % Density of chrome steel ball bearing
dia = 1.0000e-03; % Diameter of ball bearing 1mm
rad = dia/2;      % Radius of ball bearing 

volume_ball = (4/3)*pi*(rad.^3);  % ball bearing volume m^3
area_ball = 4*pi*(rad.^2);        % ball bearing area m^2
mass_ball = rho_b*volume_ball;    % ball bearing mass kg

% Fluid Density (binary mixture) Units: kg/m^3
% For liquid HFA134a and Ethanol
Temp = 19.734;             % System temperature degrees celcius
rho_propellant = 1226.3;   % R134a liquid density kg/m^3
rho_ethanol = 789.56;      % Ethanol liquid density kg/m^3

% Assume mixture viscosity value to estimate actual value
mu_propellant = 0.00020805;   % R134a liquid viscosity Pa.S
mu_ethanol = 0.0011994;       % Ethanol liquid viscosity Pa.S

x_ethanol = 0.0;             % Ethanol mass fraction
x_propellant = 1-x_ethanol;   % Propellant mass fraction

rho_mixture = (rho_ethanol*rho_propellant)/((x_ethanol*rho_propellant)+(x_propellant*rho_ethanol));
mu_mixture = (mu_ethanol*mu_propellant)/((x_ethanol*mu_propellant)+(x_propellant*mu_ethanol));

% Force calculations for weight and buoyancy
F_weight = mass_ball*g;
F_buoyance = rho_mixture*g*volume_ball;

% Inclination angle in this work
N_1 = 8;
N_2 = 10;
N_3 = 15;
N_4 = 20;
N_5 = 25;

% Weight and Buoyancy force calculations normal to surface
F_normal = (F_weight - F_buoyance)*cosd(N_1);

% Weight and Buoyancy force calculations parallel to surface
F_parallel = (F_weight - F_buoyance)*sind(N_1);

% MInimization input values
v_exp = 0.221;              % Measured velocity m/s
mu_ini = mu_mixture;   
Cd_res=1;
count=0;

while Cd_res > 10^(-9)
    
    count=count+1;
    
    % Drag coefficient calculation
    % Reynolds Number (Re)
    Re_cal = (rho_mixture*v_exp*dia)/mu_ini;

    % C_d correlations
    %Cd_cal = 2.6689 + (21.683/Re_cal) + (0.131/(Re_cal.^2)) - (10.616/(Re_cal.^0.1)) + (12.216/(Re_cal.^0.2));
    %E = 0.261*(Re_cal.^0.369) - 0.105*(Re_cal.^0.431) - (0.124/(1+((log10(Re_cal)).^2)))
    %Cd_cal = (24/Re_cal)*(10.^E); % Blakes et al
    Cd_cal = (2.25*(Re_cal.^(-0.31)) + 0.36*(Re_cal.^(0.06))).^(3.45); % Khan et al.
    
    % Friction coefficient estimation from Cf/Cd vs Re
    %Y = 0.0439*exp(0.0015*Re_cal); % Exponential Function
    Y = 2e-7*(Re_cal.^2) + 5e-5*(Re_cal) + 0.098;       % Polynomial Function
    
    % Calcualted friction coefficeint
    Cf_cal = Cd_cal*Y

    % Friction force calculation using Cf
    F_friction = Cf_cal*F_normal;
    F_drag = F_friction + F_parallel;
     
    % Drag coefficient calculation from F_drag
    Cd_new = (2*F_drag)/(rho_mixture*v_exp*v_exp*area_ball);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section neeeds further attention
    
    % Re_new from Cd_new
    Re = linspace(10,4000,500);
    Cd = (2.25*(Re.^(-0.31)) + 0.36*(Re.^(0.06))).^(3.45);
    data_cd =[Cd]';
    data_re =[Re]';
    
    % Finding a cell locaiton where difference between Cd_new and Cd is minimum
    for i=length(data_cd)
        del_cd = abs(Cd_new - data_cd(:,1));
    end
    
    data = [data_re,del_cd];
    [M, idx] = min(del_cd);
    Re_new = data(idx,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calculate the new viscosity
    mu_new = (rho_mixture*v_exp*dia)/Re_new;
    
    % Residual for Cd, Re, and viscosity
    mu_res = abs(mu_ini - mu_new);     
    Re_res = abs(Re_cal - Re_new);
    Cd_res = abs(Cd_cal - Cd_new);

    history_data(count,1) = mu_res;
    history_data(count,2) = Re_res;
    history_data(count,3) = Cd_res;
    
    Re_res
    mu_ini=mu_new

    if count > 100
        break
    end 
    count
    clear vars Re data_cd data_re 
end  






