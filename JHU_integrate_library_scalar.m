%{
Here I implement the full 27 term library described in the paper
%}

%clear;

addpath('../bin/')

%{
load('grid.mat')
load('data/NS_centerbig_data_1.mat');   %velocities
load('data/NS_centerbig_data_p_1.mat'); %pressure

p = p - mean(p, 'all'); %shift to zero mean

dx = x(2) - x(1);
dy = y(2) - y(1);
dz = z(2) - z(1);
dt = 0.1950/30;
dx_vec = [dx, dy, dz, dt];
%return
%}

%% SCALES

p = p - mean(p,'all');

U_mean = mean( sqrt(U.^2 + V.^2 + W.^2),    'all' );
U_std  = std(  sqrt(U.^2 + V.^2 + W.^2), 0, 'all' );
P_mean = mean( abs(p), 'all' );
P_std  = std(  p, 0,   'all' );

[ L_scale, T_scale ] = generate_length_and_time_scales(U, V, W, dx_vec);

%%INTEGRATION 
num_lib = 25; %purposely lower than the "true" 28
num_windows = 1000; %25*num_lib;
nw = num_windows;
dof = 1; %3 degrees of freedom for vector equation, 1 for scalar


G = zeros( dof*num_windows, num_lib );
labels = cell(num_lib, 1);
a = 1;
m = [1 1 1 1]*6;
n = [1 1 1 1]*6;

mask = 0*U+1;

corners  = zeros(4, num_windows);
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

%Derivatives
[U_y, U_x, U_z, U_t] = gradient( U, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[V_y, V_x, V_z, V_t] = gradient( V, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[W_y, W_x, W_z, W_t] = gradient( W, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[p_y, p_x, p_z, p_t] = gradient( p, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );

div = U_x + V_y + W_z;

labels{a} = "1";
G(:,a)    = gpu_integrate_nd( 0*p+1, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = 1;
ideal(a)  = 0;
a = a+1;

labels{a} = "p";
G(:,a)    = gpu_integrate_nd( p, [], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_mean;
ideal(a)  = 0;
a = a+1;

labels{a} = "\partial_t p";
G(:,a)    = gpu_integrate_nd( p, [4], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_std/T_scale;
ideal(a)  = 0;
a = a+1;

labels{a} = "\partial_t^2 p";
G(:,a)    = gpu_integrate_nd( p, [4,4], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_std/T_scale^2;
ideal(a)  = 0;
a = a+1;

labels{a} = "p^2";
G(:,a)    = gpu_integrate_nd( p.^2, [], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_mean^2;
ideal(a)  = 0;
a = a+1;

labels{a} = "p \partial_t p";
G(:,a) = 0.5*gpu_integrate_nd( p.^2, [4], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_mean*P_std/T_scale;
ideal(a)  = 0;
a = a+1;

labels{a} = "p \partial_t^2 p";
G(:,a) = gpu_integrate_nd(   p.*p_t,[4], dx_vec, corners, size_vec, m, n, [] ) ...
        -gpu_integrate_nd( p_t.*p_t, [], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_mean*P_std/T_scale^2;
ideal(a)  = 0;
a = a+1;

labels{a} = "\partial_t p \partial_t p";
G(:,a) = gpu_integrate_nd( p_t.^2, [], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_std^2 / T_scale^2;
ideal(a)  = 0;
a = a+1;

labels{a} = "\nabla^2 p";
G(:,a)    = gpu_integrate_nd( p, [1,1], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( p, [2,2], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( p, [3,3], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = P_std/L_scale^2;
ideal(a)  = 1;
a = a+1;

%{
labels{a} = "\nabla_i u_i";
G(:,a)    = gpu_integrate_nd( U, [1], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( V, [2], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( W, [3], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = U_std/L_scale;
ideal(a)  = 0;
a = a+1;

labels{a} = "\partial_t \nabla_i u_i";
G(:,a)    = gpu_integrate_nd( U, [1,4], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( V, [2,4], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( W, [3,4], dx_vec, corners, size_vec, m, n, [] );      
scales(a) = U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;
%}

labels{a} = "p \nabla^2 p";
G(:,a)    = gpu_integrate_nd( p.*p_x, [1], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( p.*p_y, [2], dx_vec, corners, size_vec, m, n, [] ) + ...
            gpu_integrate_nd( p.*p_z, [3], dx_vec, corners, size_vec, m, n, [] ) ...
           -gpu_integrate_nd( p_x.^2 + p_y.^2 + p_z.^2, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = P_mean*P_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "p \nabla_i u_i";
G(:,a)    = gpu_integrate_nd( p.*div, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = P_mean*U_std/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "p \partial_t \nabla_i u_i";
G(:,a)    = gpu_integrate_nd( p.*div, [4], dx_vec, corners, size_vec, m, n, [] ) ...
           -gpu_integrate_nd( p_t.*div, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = P_mean*U_std/T_scale/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t p \nabla_i u_i";
G(:,a)    = gpu_integrate_nd( p_t.*div, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = P_std*U_std/T_scale/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_i p \nabla_i p";
G(:,a)    = gpu_integrate_nd( p_x.^2 + p_y.^2 + p_z.^2, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = P_std^2/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \nabla_i p";
G(:,a)    = gpu_integrate_nd( p_x.*U + p_y.*V + p_z.*W, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*P_std/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t u_i \nabla_i p";
G(:,a)    = gpu_integrate_nd( p_x.*U_t + p_y.*V_t + p_z.*W_t, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_std*P_std/T_scale/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = " u_i \nabla_i \partial_t p";
G(:,a)    =  gpu_integrate_nd( p_x.*U + p_y.*V + p_z.*W, [4], dx_vec, corners, size_vec, m, n, [] ) ...
            -gpu_integrate_nd( p_x.*U_t + p_y.*V_t + p_z.*W_t, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*P_std/T_scale/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u^2";
G(:,a)    =  gpu_integrate_nd( U.^2 + V.^2 + W.^2, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean^2;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \partial_t u_i";
G(:,a)    =  gpu_integrate_nd( U.*U_t + V.*V_t + W.*W_t, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*U_std/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \partial_t^2 u_i";
G(:,a) =  gpu_integrate_nd( U.*U_t + V.*V_t + W.*W_t, [4], dx_vec, corners, size_vec, m, n, [] ) ...
         -gpu_integrate_nd( U_t.*U_t + V_t.*V_t + W_t.*W_t, [], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*U_std/T_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t u_i \partial_t u_i";
G(:,a)    =  gpu_integrate_nd( U_t.*U_t + V_t.*V_t + W_t.*W_t, [], dx_vec, corners, size_vec, m, n, [] );
scales(a)  = U_std^2 / T_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \nabla_i \nabla_j u_j";
G(:,a) =  gpu_integrate_nd( U.*div,   [1], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( V.*div,   [2], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( W.*div,   [3], dx_vec, corners, size_vec, m, n, [] ) ...
         -gpu_integrate_nd( div.*div, [ ], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \nabla^2 u_i";
U_sq = U.^2 + V.^2 + W.^2;
G(:,a) =  gpu_integrate_nd( U_sq,   [1,1], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( U_sq,   [2,2], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( U_sq,   [3,3], dx_vec, corners, size_vec, m, n, [] ) ...
         -gpu_integrate_nd( U_x.^2 + U_y.^2 + U_z.^2 + V_x.^2 + V_y.^2 + V_z.^2 + W_x.^2 + W_y.^2 + W_z.^2, [ ], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_mean*U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

%{
%Replacing this with \nabla_i \nabla_j(u_i u_j)
labels{a} = "(\nabla_i u_i)^2";
G(:,a) = gpu_integrate_nd( div.*div, [ ], dx_vec, corners, size_vec, m, n, [] );
a = a+1;
%}

labels{a} = "\nabla_i u_j \nabla_i u_j";
G(:,a) = gpu_integrate_nd( U_x.^2 + U_y.^2 + U_z.^2 + V_x.^2 + V_y.^2 + V_z.^2 + W_x.^2 + W_y.^2 + W_z.^2, [ ], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_std^2/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_i u_j \nabla_j u_i";
G(:,a) = gpu_integrate_nd( U_x.^2 + V_y.^2 + W_z.^2 ...
                         + 2*U_y.*V_x + 2*U_z.*W_x+ 2*V_z.*W_y, [ ], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_std^2/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = '\nabla_i \nabla_j (u_i u_j)';
G(:,a) =  gpu_integrate_nd( U.*U,   [1,1], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( V.*V,   [2,2], dx_vec, corners, size_vec, m, n, [] ) ...
         +gpu_integrate_nd( W.*W,   [3,3], dx_vec, corners, size_vec, m, n, [] ) ...
         +2*gpu_integrate_nd( U.*V,   [1,2], dx_vec, corners, size_vec, m, n, [] ) ...
         +2*gpu_integrate_nd( U.*W,   [1,3], dx_vec, corners, size_vec, m, n, [] ) ...
         +2*gpu_integrate_nd( V.*W,   [2,3], dx_vec, corners, size_vec, m, n, [] );
scales(a) = U_std^2/L_scale^2;
ideal(a) = 1;
a = a+1;



%Lastly, normalize by the integral of unity to avoid grid effects
norm_vec = gpu_integrate_nd( 0*U + 1, [], dx_vec, corners, size_vec, m, n, [] );      
G = G/norm( norm_vec );

G = G./scales;

return
%Remove troublesome terms
bad = vecnorm(G) > 1e-4;
G = G(:,bad);
labels = labels(bad);
ideal = ideal(bad);
scales = scales(bad);




function [ L_scale, T_scale ] = generate_length_and_time_scales(U, V, W, dx_vec)
  dx = dx_vec(1);
  dy = dx_vec(2);
  dz = dx_vec(3);
  dt = dx_vec(4);
  
  [U_y, U_x, U_z, U_t] = gradient( U, dx, dy, dz, dt );
  [V_y, V_x, V_z, V_t] = gradient( V, dx, dy, dz, dt );
  [W_y, W_x, W_z, W_t] = gradient( W, dx, dy, dz, dt );

  %Calculate the average magnitude of vorticity as a time scale
  T_scale = 1/mean( (U_y - U_z).^2 + (V_x - V_z).^2 + (W_x - W_y).^2, 'all' );
  L_scale = mean( sqrt(U.^2 + V.^2 + W.^2), 'all' )*T_scale;
end