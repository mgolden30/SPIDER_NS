%{
Matt's take on SPIDER for the JHU turbulence data
%}
clear;

addpath('../bin/')

load('data/NS_centerbig_data_1.mat');   %velocities
load('data/NS_centerbig_data_p_1.mat'); %pressure

dx = x(2) - x(1);
dy = y(2) - y(1);
dz = z(2) - z(1);
dt = 0.1950/30;
dx_vec = [dx, dy, dz, dt];





%% INTEGRATION 
num_lib = 6;
num_windows = 50*num_lib;
nw = num_windows;
dof = 3; %3 degrees of freedom for vector equation

G = zeros( dof*num_windows, num_lib );
labels = cell(num_lib, 1);
a = 1;
m = [1 1 1 1]*6;
n = [1 1 1 1]*6;

mask = 0*U+1;

corners  = zeros(4, num_windows);
%size_vec = 1.5*[10 10 10 10];
size_vec = [27 32 33 38]; %directly from SPIDER paper
size_vec = round(size_vec); %make sure it's integers

L = 3;
Lx = size(U,1); Ly = size(U,2); Lz = size(U,3); Lt = size(U,4);
for i=1:num_windows
  for j=1:4
    corners(j,i) = randi( size(U,j) - size_vec(j) - 2*L ) + L;
  end
end

%Search only at the middle of the channel
%corners(2,:) = round((Ly - size_vec(2))/2);

labels{a} = "\partial_t u_i";
G(:,a) = gpu_integrate_curl( U, V, W, [4], dx_vec, corners, size_vec, m, n );      
a = a+1;

labels{a} = "u_i";
G(:,a) = gpu_integrate_curl( U, V, W, [], dx_vec, corners, size_vec, m, n );      
a = a+1;


labels{a} = "\nabla_j(u_j u_i)";
G(:,a) = gpu_integrate_curl( U.*U, U.*V, U.*W, [1], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_curl( V.*U, V.*V, V.*W, [2], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_curl( W.*U, W.*V, W.*W, [3], dx_vec, corners, size_vec, m, n );
a = a+1;


labels{a} = "\nabla^2 u_i";
G(:,a) = gpu_integrate_curl( U, V, W, [1,1], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_curl( U, V, W, [2,2], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_curl( U, V, W, [3,3], dx_vec, corners, size_vec, m, n ); 
a = a+1;


labels{a} = "u^2 u_i";
U_sq = U.^2 + V.^2 + W.^2;
G(:,a) = gpu_integrate_curl( U_sq.*U, U_sq.*V, U_sq.*W, [], dx_vec, corners, size_vec, m, n ); 
a = a+1;


labels{a} = "\partial_t^2 u_i";
G(:,a) = gpu_integrate_curl( U, V, W, [4,4], dx_vec, corners, size_vec, m, n ); 
a = a+1;

%{
labels{a} = "\nabla_i \nabla_j u_j";
G(:,a) = gpu_integrate_curl( U, V, W, [4,4], dx_vec, corners, size_vec, m, n ); 
a = a+1;
%}


function vals = gpu_integrate_curl( U, V, W, derivs, dx_vec, corners, size_vec, m, n )
  %{
  This function performs integration including a cross product so that
  pressure data isn't needed to find the Navier-Stokes equation
  
  INPUT:
  U,V,W - x,y,z components of a vector field
  
  OUTPUT:
  vals - 
  %}

  vals_x = gpu_integrate_nd( W, [derivs,2], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( V, [derivs,3], dx_vec, corners, size_vec, m, n, [] );
  vals_y = gpu_integrate_nd( U, [derivs,3], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( W, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
  vals_z = gpu_integrate_nd( V, [derivs,1], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( U, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
  
  vals = [vals_x; 
          vals_y; 
          vals_z];
end