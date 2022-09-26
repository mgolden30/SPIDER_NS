%{
SPIDER script for Channelflow data
%}

clear

addpath('./bin/')

%folder = "Re1e3/Ly_140/data_nl_rot/";
folder = "Re1e4_different_nl/skew/data/";
%folder = "varying_RE/Re_6400/data/"
%folder = "Re1e3/Ly_220/data/"
%folder = "varying_RE/Re_800/data/";

%Automate read-in to prevent headaches
%files = {dir(folder).name}; %create a cell containing all files in the folder
%files{ ~contains(files,'u')   } = []; %remove any files not containing a u
%files{ ~contains(files,'.nc') } = []; %remove any files not containing .nc
%N = numel(files); %count the number of reamining files

dt = 0.01;
N = 100;

Ly = 140;
U = zeros([106, Ly, 42, N]);
V = U; 
W = U;
p = U;

file = ncinfo(folder + "u0.000.nc")
%file.Variables.Name


for i=1:N
  file  = folder + sprintf("u%.3f.nc", dt*(i-1) );
  pfile = folder + sprintf("p%.3f.nc", dt*(i-1) ); 
  
  X = ncread( file, 'X' );
  Y = ncread( file, 'Y' );
  Z = ncread( file, 'Z' );
 
  Lx = X(end) + X(2); %Length (physical units not grid points) in x
  Lz = Z(end) + Z(2); %same for z
  
  Vx = ncread(  file, 'Velocity_X'  );
  Vy = ncread(  file, 'Velocity_Y'  );
  Vz = ncread(  file, 'Velocity_Z'  );
  P  = ncread(  pfile, 'Component_0' );
  
  %X and Z are fine, but Y isn't linearly spaced
  Y_linear = linspace(-1, 1, Ly);  
  for xx = 1:size(U,1)
    for zz = 1:size(U,3)
      U(xx,:,zz,i) = interp1( Y, Vx(xx,:,zz), Y_linear, 'makima' ); 
      V(xx,:,zz,i) = interp1( Y, Vy(xx,:,zz), Y_linear, 'makima' ); 
      W(xx,:,zz,i) = interp1( Y, Vz(xx,:,zz), Y_linear, 'makima' );  
      p(xx,:,zz,i) = interp1( Y,  P(xx,:,zz), Y_linear, 'makima' );
    end
  end
end

%Now read in grid spacings
dy = Y_linear(2) - Y_linear(1); %We've changed the spacing in y!
dx = X(2) - X(1);
dz = Z(2) - Z(1);
dx_vec = [dx, dy, dz, dt];