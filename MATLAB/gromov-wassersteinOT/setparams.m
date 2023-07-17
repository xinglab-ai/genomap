%%
% Main parameters.


% if not(exist('dist_mode'))
%     dist_mode = 'geodesic';
%     dist_mode = 'euclidean';
% end
if not(exist('init')) % which cloud is used for initialization
    init = 1;
end
% loss being used 
%if not(exist('gw_loss'))
    gw_loss = 'kl';
    %gw_loss = 'l2';
    sigma = .8;
%end
if strcmp(gw_loss(1:2),'kl') 
    if not(isempty(gw_loss(3:end)))
        sigma = str2num(gw_loss(3:end))/10;
    end
    gw_loss = 'kl';
end
% % embedding method
% mdsmethod = 'classical';
% mdsmethod = 'smacof';

%% 
% rep creation





%%
% Other parameters.

% size of the input clouds
N0 = 400; 
N0 = 1000; 
% parameter for geodesic distances computations.
t_Varadhan = 1e-2;  % time step for Varadhan formula

options.gw_loss = gw_loss;
% switch from matrix distance to kernel
switch gw_loss
    case 'l2'
        Kembed = @(d)d;
        iKembed = @(K)K;
    case 'kl'
        if not(exist('sigma'))
            sigma = 1;
        end
        Kembed = @(d)exp(-(d/sigma).^2);
        iKembed = @(K)sqrt(max(-log(K),0))*sigma;
    otherwise 
        error('Unknorn loss.');
end

%%
% Helpers.

dotp = @(x,y)sum(x(:).*y(:));
mynorm = @(x)norm(x(:));
normalize = @(x)x/sum(x(:));

%%
% Test for one GW computation.

% N=400
if strcmp(gw_loss, 'l2')
    epsilon = .1/14; % 2-D shapes L2, N=400
    epsilon = .1/20;
    epsilon = .1/15;
else
    epsilon = .1/40; % 2-D shapes KL
    epsilon = .1/10; % sigma=.8, ok
    epsilon = .1/5;
end

% epsilon = 0; % exact-EMD
options.niter = 40; 
options.niter_sinkhorn = 100;
options.tol_sinkhorn = 1e-6; % tolerance on marginals for Sinkhorn
options.tol_gw = 1e-4; % early stopping for GW
options.gamma_init = [];
options.verb = 1;

