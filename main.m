%%
%clear
%load('for_matt/Re_400_G.mat')

%% Combinatoric Search
addpath('../channelflow_SPIDER/');
number_of_terms = 2;
[models, etas, cs ] = combinatoric_search( G, labels, number_of_terms);
if numel(models) > 10
  models(1:10)
else
  models
end
return


%% Reverse regression 2
clf

starting_model = cs(46,:);
starting_model = 0*starting_model;
starting_model( [18, 35]) = 1;
%starting_model(:) = 0;
%starting_model(18) = 1;
%starting_model = ideal;
starting_model = cs_ave(:,3);

number_of_subsamples = 100; %Do parameter estimation 100 times to estimate uncertainty
fraction_of_windows_in_subsample = 1/2; %use half of windows randomly each time we estimate parameters

res1 = @(G,c) norm(G*c)/norm(c);
res2 = @(G,c) norm(G*c)/norm(c)/norm(G);
res3 = @(G,c) norm(G*c)/max( vecnorm(G*diag(c)) );
residual_functions = {res1, res2, res3}; %residuals to compute during sparsification
index = 3;
res_func_for_sparsification = residual_functions{index};
[res_ave, res_std, cs_ave, cs_std] = reverse_regression2( G, starting_model, ...
                                                                   number_of_subsamples, ...
                                                                   fraction_of_windows_in_subsample, ...
                                                                   residual_functions, ...
                                                                   res_func_for_sparsification );

                                                          
%Plot results                                                  
%h = errorbar(res_ave(1,:), res_std(1,:), 'o' );
fs = 16; %Font Size
lim = size(res_ave,2);
h = scatter( 1:lim, res_ave(index,1:lim), 'o', 'filled' );
%{
hold on
for i=2:size(res_ave,1)
  if i==2
    continue 
  end
  %h = errorbar(res_ave(i,:), res_std(i,:), 'o' );
  h = plot( res_ave() )
end
hold off
%}
set(get(h,'Parent'), 'YScale', 'log');

%ylim([0 1]);
xlabel('K', 'interpreter', 'latex', 'FontSize', fs);
ylabel('$\zeta$', 'interpreter', 'latex', 'FontSize', fs );
h1 = get(gca,'YLabel');
%set(h1,'Position', get(h1,'Position') - [0.4 -0.15 0])
xticks(1:numel(res_ave))
xlim( [0.5, size(res_ave,2)+0.5] );
set(gcf,'color','w');
pbaspect([2 1 1])

%xline( sum( starting_model ~= 0) );
%legend({'|Gc|', ...
    %'|Gc|/|G|/|c|', ...
%    '\eta'})


%% Print a model of interest
interest = 2;

c = cs_ave(:,interest);
c = c./scales';

%c = cs(2,:);
temp = find(c);
normalization = c(temp(1)); %normalize by the first nonzero element 
%normalization = max( vecnorm(G*diag(c)) ); %normalize by the first nonzero element 

str = "\zeta = " + res_ave(index,interest) + ", \quad   ";
for i = 1:numel(c)
  if( c(i) == 0 )
    continue; 
  end
  
  str = str + " + " + c(i)/normalization + "  " + labels{i};
end
str = str + " = 0"




%% Making a figure for Lambda-Omega
starting_model = cs(1,:);

number_of_subsamples = 100; %Do parameter estimation 100 times to estimate uncertainty
fraction_of_windows_in_subsample = 1/2; %use half of windows randomly each time we estimate parameters

res1 = @(G,c) norm(G*c);
res2 = @(G,c) norm(G*c)/norm(c)/norm(G);
res3 = @(G,c) norm(G*c)/max( vecnorm(G*diag(c)) );
residual_functions = {res1, res2, res3}; %residuals to compute during sparsification
res_func_for_sparsification = res1;
[res_ave, res_std, cs_ave, cs_std] = reverse_regression2( G, starting_model, ...
                                                                   number_of_subsamples, ...
                                                                   fraction_of_windows_in_subsample, ...
                                                                   residual_functions, ...
                                                                   res_func_for_sparsification );

path_x = 2:15;
path_y = res_ave(1, path_x);


starting_model = cs_ave(:,11);
[res_ave, res_std, cs_ave, cs_std] = reverse_regression2( G, starting_model, ...
                                                                   number_of_subsamples, ...
                                                                   fraction_of_windows_in_subsample, ...
                                                                   residual_functions, ...
                                                                   res_func_for_sparsification );

path_x = [path_x, 11:-1:5];
path_y = [path_y, res_ave(1, 11:-1:5 )];

%Plot results
fs = 16; %Font Size
lim = size(res_ave,2);
h = scatter( path_x, path_y, 'o', 'filled', 'MarkerFaceColor', 'black' );
hold on
  plot(path_x, path_y, 'Color', 'black');
hold off

set(get(h,'Parent'), 'YScale', 'log');

%ylim([0 1]);
xlabel('K', 'interpreter', 'latex', 'FontSize', fs);
ylabel('$|Gc|$', 'interpreter', 'latex', 'FontSize', fs );
h1 = get(gca,'YLabel');
%set(h1,'Position', get(h1,'Position') - [0.4 -0.15 0])
xticks(1:numel(res_ave))
xlim( [0.5, size(res_ave,2)+0.5] );
set(gcf,'color','w');
pbaspect([2 1 1])

%xline( sum( starting_model ~= 0) );
%legend({'|Gc|', ...
    %'|Gc|/|G|/|c|', ...
%    '\eta'})
hold on
%scatter( path_x([11,18]), path_y([11,18]), 's', 'MarkerEdgeColor', 'red',   'LineWidth', 1.5, 'SizeData', 150 );
scatter( path_x([17]), path_y([17]), 's', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'SizeData', 150 );
scatter( path_x([15]), path_y([15]), 's', 'MarkerEdgeColor', 'blue',  'LineWidth', 1.5, 'SizeData', 150 );
hold off
xlim([1 13])
legend({'','','O(2) symmetric', 'symmetry broken'})


%% Sequential least squares (deprecated)
G_fake = 1; %Not using syntehtic data right now
threshold = 0.1;
[models, magnitudes, is_model_identity] = list_all_models( G, G_fake, labels, threshold );
models


%% Compare

%1 V -0.8754  C + 1.9087  C^2 + -2.0772  V*C = 0

vidObj = VideoWriter('constraint.avi');
open(vidObj);
for T = 1:3:1000
tiledlayout(1,2);

%T = 100;

nexttile
imagesc(V(:,:,T))
axis square
colorbar
caxis([0 1])
title('V');
caxis()

nexttile
imagesc( 0.8754*C(:,:,T) - 1.9087*C(:,:,T).^2 + 2.0772*V(:,:,T).*C(:,:,T)  );
axis square
colorbar
caxis([0 1]);
title('0.8754 C - 1.9087 C^2 + 2.0772 V*C');

set(gcf, 'Color', 'w')
drawnow

       currFrame = getframe(gcf);
       writeVideo(vidObj,currFrame);
end
    close(vidObj);

