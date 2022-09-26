%{
a 21-term library generated by careful consideration of the fundamental
tensors. All terms more than quadratic in physical fields/ containing more
than two derivatives have been removed for simplicity
%}

addpath('../bin/')

global symm_tf 
symm_tf = false;

%{
load('grid.mat')
load('data/NS_centerbig_data_1.mat');   %velocities
load('data/NS_centerbig_data_p_1.mat'); %pressure


dx = x(2) - x(1);
dy = y(2) - y(1);
dz = z(2) - z(1);
dt = 0.1950/30;
dx_vec = [dx, dy, dz, dt];
%}

%% SCALES

p = p - mean(p, 'all'); %Do this

U_mean = mean( sqrt(U.^2 + V.^2 + W.^2),    'all' );
U_std  = std(  sqrt(U.^2 + V.^2 + W.^2), 0, 'all' );
P_mean = mean( abs(p), 'all' );
P_std  = std(  p, 0,   'all' );

[ L_scale, T_scale ] = generate_length_and_time_scales(U, V, W, dx_vec);



%
num_lib = 21;
num_windows = round(1000/3); %25*num_lib;
nw = num_windows;
dof = 3; %3 degrees of freedom for vector equation
if symm_tf
  dof = 5; 
end
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

%Compute derivatives
[U_x, U_y, U_z, U_t] = gradient( U, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[V_x, V_y, V_z, V_t] = gradient( V, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[W_x, W_y, W_z, W_t] = gradient( W, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );
[p_x, p_y, p_z, p_t] = gradient( p, dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4) );

div = U_x + V_y + W_z;



a = 1;

tic

labels{a} = "\nabla^i p";
G(:,a)    = gpu_integrate_gradient( p, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std/L_scale;
ideal(a) = 1;
a = a+1;

labels{a} = "\partial_t \nabla^i p";
G(:,a) = gpu_integrate_gradient( p, [4], dx_vec, corners, size_vec, m, n );
scales(a) = P_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i";
G(:,a) = gpu_integrate_vector( U, V, W, [], dx_vec, corners, size_vec, m, n );      
scales(a) = U_mean;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t u_i";
G(:,a) = gpu_integrate_vector( U, V, W, [4], dx_vec, corners, size_vec, m, n );      
scales(a) = U_std/T_scale;
ideal(a) = 1;
a = a+1;

labels{a} = "\partial_t^2 u_i";
G(:,a)    = gpu_integrate_vector( U, V, W, [4,4], dx_vec, corners, size_vec, m, n );      
scales(a) = U_std/T_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "p\nabla_i p";
G(:,a)    = gpu_integrate_gradient( 0.5*p.^2, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_mean*P_std/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "p \nabla_i \partial_t p";
G(:,a)    = gpu_integrate_gradient( p.*p_t, [], dx_vec, corners, size_vec, m, n ) ...
           -gpu_integrate_vector( p_x.*p_t, p_y.*p_t, p_z.*p_t, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_mean*P_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "p u_i";
G(:,a)    = gpu_integrate_vector( p.*U, p.*V, p.*W, [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_mean*U_mean;
ideal(a) = 0;
a = a+1;

labels{a} = "p \partial_t u_i";
G(:,a)    = gpu_integrate_vector( p.*U_t, p.*V_t, p.*W_t, [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_mean*U_std/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "p \partial_t^2 u_i";
G(:,a)    = gpu_integrate_vector( p.*U_t, p.*V_t, p.*W_t, [4], dx_vec, corners, size_vec, m, n ) ...
           -gpu_integrate_vector( p_t.*U_t, p_t.*V_t, p_t.*W_t,  [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_mean*U_std/T_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_i p \partial_t p";
G(:,a)    = gpu_integrate_vector( p_x.*p_t, p_y.*p_t, p_z.*p_t, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std^2/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t p u_i";
G(:,a)    = gpu_integrate_vector( p_t.*U, p_t.*V, p_t.*W, [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_std*U_mean/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t p \partial_t u_i";
G(:,a)    = gpu_integrate_vector( p_t.*U_t, p_t.*V_t, p_t.*W_t, [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_std*U_std/T_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t^2 p u_i";
G(:,a)    = gpu_integrate_vector( p_t.*U, p_t.*V, p_t.*W, [4], dx_vec, corners, size_vec, m, n ) ...
           -gpu_integrate_vector( p_t.*U_t, p_t.*V_t, p_t.*W_t,  [], dx_vec, corners, size_vec, m, n );      
scales(a) = P_std*U_mean/T_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla^2 u_i";
G(:,a)    = gpu_integrate_vector( U, V, W, [1,1], dx_vec, corners, size_vec, m, n ) ...
          + gpu_integrate_vector( U, V, W, [2,2], dx_vec, corners, size_vec, m, n ) ...
          + gpu_integrate_vector( U, V, W, [3,3], dx_vec, corners, size_vec, m, n ); 
scales(a) = U_std/L_scale/L_scale;
ideal(a) = 1;
a = a+1;

labels{a} = "\nabla^i \nabla_j u^j";
G(:,a) = gpu_integrate_gradient( div, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "p \nabla^2 u_i";
G(:,a)    = gpu_integrate_vector( p.*U_x, p.*V_x, p.*W_x, [1], dx_vec, corners, size_vec, m, n ) ...
          + gpu_integrate_vector( p.*U_y, p.*V_y, p.*W_y, [2], dx_vec, corners, size_vec, m, n ) ...
          + gpu_integrate_vector( p.*U_z, p.*V_z, p.*W_z, [3], dx_vec, corners, size_vec, m, n ) ...
          - gpu_integrate_vector( p_x.*U_x + p_y.*U_y + p_z.*U_z,         p_x.*V_x + p_y.*V_y + p_z.*V_z,     p_x.*W_x + p_y.*W_y + p_z.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_mean*U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "p \nabla_i \nabla_j u_j";
G(:,a)    = gpu_integrate_gradient( p.*div, [], dx_vec, corners, size_vec, m, n ) ...
           -gpu_integrate_vector( p_x.*div, p_y.*div, p_z.*div, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_mean*U_std/L_scale^2;
a = a+1;

labels{a} = "\nabla_i p \nabla_j u_j";
G(:,a)    = gpu_integrate_vector( p_x.*div, p_y.*div, p_z.*div, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std*U_std/L_scale/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_j p \nabla_i u_j";
G(:,a) = gpu_integrate_vector( p_x.*U_x + p_y.*V_x + p_z.*W_x, ...
                               p_x.*U_y + p_y.*V_y + p_z.*W_y, ...
                               p_x.*U_z + p_y.*V_z + p_z.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std*U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_j p \nabla_j u_i";
G(:,a) = gpu_integrate_vector( p_x.*U_x + p_y.*U_y + p_z.*U_z, ...
                               p_x.*V_x + p_y.*V_y + p_z.*V_z, ...
                               p_x.*W_x + p_y.*W_y + p_z.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std*U_std/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla_i \nabla_j p u_j";
G(:,a) = gpu_integrate_gradient( U.*p_x + V.*p_y + W.*p_z, [], dx_vec, corners, size_vec, m, n ) ...
        -gpu_integrate_vector( p_x.*U_x + p_y.*V_x + p_z.*W_x, ...
                               p_x.*U_y + p_y.*V_y + p_z.*W_y, ...
                               p_x.*U_z + p_y.*V_z + p_z.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std*U_mean/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "\nabla^2 p u_i";
G(:,a) = gpu_integrate_vector( p_x.*U, p_x.*V, p_x.*W, [1], dx_vec, corners, size_vec, m, n ) ...
        +gpu_integrate_vector( p_y.*U, p_y.*V, p_y.*W, [2], dx_vec, corners, size_vec, m, n ) ...
        +gpu_integrate_vector( p_z.*U, p_z.*V, p_z.*W, [3], dx_vec, corners, size_vec, m, n ) ...
        -gpu_integrate_vector( p_x.*U_x + p_y.*U_y + p_z.*U_z, ...
                               p_x.*V_x + p_y.*V_y + p_z.*V_z, ...
                               p_x.*W_x + p_y.*W_y + p_z.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = P_std*U_mean/L_scale^2;
ideal(a) = 0;
a = a+1;

labels{a} = "u_i \nabla_j u_j";
G(:,a) = gpu_integrate_vector( U.*div, V.*div, W.*div, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u_j \nabla^i u^j";
G(:,a) = gpu_integrate_gradient( 0.5*(U.^2 + V.^2 + W.^2), [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale;
a = a+1;

%instead of u^j \nabla_j u^i
labels{a} = "\nabla_j(u_j u_i)";
G(:,a) = gpu_integrate_vector( U.*U, U.*V, U.*W, [1], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_vector( V.*U, V.*V, V.*W, [2], dx_vec, corners, size_vec, m, n ) ...
       + gpu_integrate_vector( W.*U, W.*V, W.*W, [3], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale;
ideal(a) = 1;
a = a+1;

labels{a} = "u^i \nabla_j \partial_t u_j";
G(:,a) = gpu_integrate_vector( U.*div, V.*div, W.*div, [4], dx_vec, corners, size_vec, m, n ) ...
        -gpu_integrate_vector( U_t.*div, V_t.*div, W_t.*div, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u_j \nabla_i \partial_t u_j";
G(:,a) = gpu_integrate_gradient( U.*U_t + V.*V_t + W.*W_t, [], dx_vec, corners, size_vec, m, n ) ...
        -gpu_integrate_vector( U_t.*U_x + V_t.*V_x + W_t.*W_x, ...
                               U_t.*U_y + V_t.*V_y + W_t.*W_y, ...
                               U_t.*U_z + V_t.*V_z + W_t.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "u^j \nabla_j \partial_t u_i";
G(:,a) = gpu_integrate_vector( U.*U_x + V.*U_y + W.*U_z, U.*V_x + V.*V_y + W.*V_z, U.*W_x + V.*W_y + W.*W_z, [4], dx_vec, corners, size_vec, m, n ) ...
        -gpu_integrate_vector( U_t.*U_x + V_t.*U_y + W_t.*U_z, U_t.*V_x + V_t.*V_y + W_t.*V_z, U_t.*W_x + V_t.*W_y + W_t.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t u_i \nabla_j u_j";
G(:,a) = gpu_integrate_vector( U_t.*div, V_t.*div, W_t.*div, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_std*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;


labels{a} = "\partial_t u_j \nabla_i u_j";
G(:,a) = gpu_integrate_vector( U_t.*U_x + V_t.*V_x + W_t.*W_x, ...
                               U_t.*U_y + V_t.*V_y + W_t.*W_z, ...
                               U_t.*U_z + V_t.*V_z + W_t.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;

labels{a} = "\partial_t u_j \nabla_j u_i";
G(:,a) = gpu_integrate_vector( U_t.*U_x + V_t.*U_y + W_t.*U_z, U_t.*V_x + V_t.*V_y + W_t.*V_z, U_t.*W_x + V_t.*W_y + W_t.*W_z, [], dx_vec, corners, size_vec, m, n );
scales(a) = U_mean*U_std/L_scale/T_scale;
ideal(a) = 0;
a = a+1;






%Lastly, normalize by the integral of unity to avoid grid effects
norm_vec = gpu_integrate_nd( 0*U + 1, [], dx_vec, corners, size_vec, m, n, [] );      
G = G/norm( norm_vec );

G = G./scales;

%Remove troublesome terms
bad = vecnorm(G) > 1e-4;
G = G(:,bad);
labels = labels(bad);
ideal = ideal(bad);
scales = scales(bad);

toc


function vals = gpu_integrate_vector( U, V, W, derivs, dx_vec, corners, size_vec, m, n )
  %{
  
  INPUT:
  U,V,W - x,y,z components of a vector field
  
  OUTPUT:
  vals - 
  %}

  vals_x = gpu_integrate_nd( U, derivs, dx_vec, corners, size_vec, m, n, [] );
  vals_y = gpu_integrate_nd( V, derivs, dx_vec, corners, size_vec, m, n, [] );
  vals_z = gpu_integrate_nd( W, derivs, dx_vec, corners, size_vec, m, n, [] );

  %{
  vals_x = gpu_integrate_nd( W, [derivs,2], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( V, [derivs,3], dx_vec, corners, size_vec, m, n, [] );
  vals_y = gpu_integrate_nd( U, [derivs,3], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( W, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
  vals_z = gpu_integrate_nd( V, [derivs,1], dx_vec, corners, size_vec, m, n, [] ) ...
         - gpu_integrate_nd( U, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
  %}
  
  vals = [vals_x; 
          vals_y; 
          vals_z];
      
  global symm_tf
  if symm_tf
    A_xx = gpu_integrate_nd( U, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
    A_xy = gpu_integrate_nd( U, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
    A_xz = gpu_integrate_nd( U, [derivs,3], dx_vec, corners, size_vec, m, n, [] );
    
    A_yx = gpu_integrate_nd( V, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
    A_yy = gpu_integrate_nd( V, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
    A_yz = gpu_integrate_nd( V, [derivs,3], dx_vec, corners, size_vec, m, n, [] );

    A_zx = gpu_integrate_nd( W, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
    A_zy = gpu_integrate_nd( W, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
    A_zz = gpu_integrate_nd( W, [derivs,3], dx_vec, corners, size_vec, m, n, [] );

    tr = A_xx + A_yy + A_zz;
  
    vals = [A_xx - tr/3;
            (A_xy + A_yx)/2;
            (A_xz + A_zx)/2;
            A_yy - tr/3;
            (A_yz + A_zy)/2;];
  end
end

function vals = gpu_integrate_gradient( p, derivs, dx_vec, corners, size_vec, m, n )
  %{
  
  INPUT:
  U,V,W - x,y,z components of a vector field
  
  OUTPUT:
  vals - 
  %}

  use_curl = 0; %1 for use curl, 0 for not

  vals_x = gpu_integrate_nd( p, [derivs,1], dx_vec, corners, size_vec, m, n, [] );
  vals_y = gpu_integrate_nd( p, [derivs,2], dx_vec, corners, size_vec, m, n, [] );
  vals_z = gpu_integrate_nd( p, [derivs,3], dx_vec, corners, size_vec, m, n, [] );
         
  vals = [vals_x; 
          vals_y; 
          vals_z];
      
  vals = (1-use_curl)*vals;
  
  global symm_tf;
  if symm_tf
    A_xx = gpu_integrate_nd( p, [derivs,1,1], dx_vec, corners, size_vec, m, n, [] );
    A_xy = gpu_integrate_nd( p, [derivs,1,2], dx_vec, corners, size_vec, m, n, [] );
    A_xz = gpu_integrate_nd( p, [derivs,1,3], dx_vec, corners, size_vec, m, n, [] );
    
    %A_yx = gpu_integrate_nd( p, [derivs,2,1], dx_vec, corners, size_vec, m, n, [] );
    A_yx = A_xy;
    A_yy = gpu_integrate_nd( p, [derivs,2,2], dx_vec, corners, size_vec, m, n, [] );
    A_yz = gpu_integrate_nd( p, [derivs,2,3], dx_vec, corners, size_vec, m, n, [] );

    %A_zx = gpu_integrate_nd( p, [derivs,3,1], dx_vec, corners, size_vec, m, n, [] );
    A_zx = A_xz;
    A_zy = A_yz;
    %    A_zy = gpu_integrate_nd( p, [derivs,3,2], dx_vec, corners, size_vec, m, n, [] );
    A_zz = gpu_integrate_nd( p, [derivs,3,3], dx_vec, corners, size_vec, m, n, [] );

    tr = A_xx + A_yy + A_zz;
  
    vals = [A_xx - tr/3;
            (A_xy + A_yx)/2;
            (A_xz + A_zx)/2;
            A_yy - tr/3;
            (A_yz + A_zy)/2;];
  end
end



function [ L_scale, T_scale ] = generate_length_and_time_scales(U, V, W, dx_vec)
  dx = dx_vec(1);
  dy = dx_vec(2);
  dz = dx_vec(3);
  dt = dx_vec(4);
  
  [U_x, U_y, U_z, U_t] = gradient( U, dx, dy, dz, dt );
  [V_x, V_y, V_z, V_t] = gradient( V, dx, dy, dz, dt );
  [W_x, W_y, W_z, W_t] = gradient( W, dx, dy, dz, dt );

  %Calculate the average magnitude of vorticity as a time scale
  T_scale = 1/mean( (U_y - U_z).^2 + (V_x - V_z).^2 + (W_x - W_y).^2, 'all' );
  L_scale = mean( sqrt(U.^2 + V.^2 + W.^2), 'all' )*T_scale;
end