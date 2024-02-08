function model = NPSVCPP_train(X, Y, param, varargin)
% Train kernelized NPSVC++
%
%    input:
%          X: data matrix, with ROW as sample
%          Y: label, from 1~c.
%      param: include c, r, kernel parameters.
%    output:
%          model: the learned model
%
%    Written by Apple Zhang, 2023.

p = inputParser;
p.KeepUnmatched(true);
p.addParameter("MAXITER", 10, ...
    @(x) assert(isnumeric(x) && x > 0, ...
    "maximum iteration should be a positive number."));
p.addParameter("BETA", .75, ...
    @(x) assert(isnumeric(x) && x >= 0 && x <= 1, ...
    "beta should be a number in [0,1]"));
p.addParameter("EPSILON", 1e-3, ...
    @(x) assert(isnumeric(x) && x > 0, ...
    "epsilon should be a positive number"));
p.addParameter("verbose", false, ...
    @(x) assert(islogical(x), ...
    "Verbose option can only be true or false."));
p.addParameter("LearningRate", 0.1, ...
    @(x) assert(isnumeric(x) && x > 0, ...
    "Leraning rate should be positive number."));
p.addParameter("MultiBirth", true, ...
    @(x) assert(islogical(x), ...
    "The option multi-birth should be logical."));
p.addParameter("manualKernel", false, ...
    @(x) assert(islogical(x), ...
    "The manual kernel option should be logical"));
p.addParameter("lr", 0.1, ...
    @(x) assert(isscalar(x) && x > 0, ...
    "The learning rate should be a scalar."));

parse(p, varargin{:});
opt = p.Results;

EPSILON = opt.EPSILON;
BETA = opt.BETA;
LR   = opt.lr;
MAX_ITER = round(opt.MAXITER);

[n, m] = size(X);
uY = unique(Y);
k  = numel(uY);

% default parameters
if ~isfield(param, "d"),     param.d  = min(2*k, m-1); end
if ~isfield(param, "mu"),    param.mu = 2 / (n-1);     end
if ~isfield(param, "r1"),    param.r1 = 1/k;           end
if ~isfield(param, "r2"),    param.r2 = 1/k;           end
if ~isfield(param, "c"),     param.c = 1;              end
if ~isfield(param, "gamma"), param.gamma = .1;         end

d  = param.d;
mu = param.mu;
r1 = param.r1;
r2 = param.r2;
c  = param.c;
gamma = param.gamma;
G =  param.G;


% validate kernel param
assert(ismember(lower(param.kernel.kernelType), ...
    ["linear", "gaussian", "polynomial", "polyplus"]));

% check linear kernel
isLinearKernel = strcmpi(param.kernel.kernelType, 'linear');

if isLinearKernel
    assert(d <= m, "NPSVM++:DimensionError", ...
    "Target dimension d should be not larger than the dimension M under linear setting.");
else
    assert(d <= n, "NPSVM++:DimensionError", ...
    "Target dimension d should be not larger than the sample size N under nonlinear setting.");
end

% objective function recorder
obj.disloss   = zeros(1+MAX_ITER, k);
obj.simloss   = zeros(1+MAX_ITER, k);
obj.w2reg     = zeros(1+MAX_ITER, k);
obj.v2reg     = zeros(1+MAX_ITER, k);
obj.Preg      = zeros(1+MAX_ITER, 1);
obj.total     = zeros(1+MAX_ITER, k);
obj.tau       = zeros(1+MAX_ITER, k);
obj.dualobj   = zeros(1+MAX_ITER, 1);
obj.primalobj = zeros(1+MAX_ITER, 1);
obj.valAcc    = zeros(1+MAX_ITER, 1);
qpoptions = optimoptions('quadprog', 'Display', 'none');

% construct model
npsvcpp.uY = uY;

% kernel
if isLinearKernel
    Psi = X;
else
    if opt.manualKernel
        K = X + EPSILON * eye(n);
    else
        [Y, ind] = sort(Y, "ascend");
        X = X(ind, :);
        [K, param.kernel] = constructKernel(X, [], param.kernel);
        K = (K+K')/2 + EPSILON * eye(n);
    end
    % Decomposition of kernel
    Psi = chol(K);

    % in the nonlinear case, m is replaced by n.
    m = n;
end

% initialize
rng(110);
PsiB = randn(m, k);
V    = randn(d, k);
PsiA = orth(randn(m, d));
Omega = zeros(m*d, k);

% uniform init for the class-weight
tau  = ones(k,1) / k;

% normalized laplacian
dd = sum(G);
L = speye(n) - G ./ sqrt(dd'*dd);
KLK = Psi*L*Psi';

[KQ, PsiMPsi, KMPsi] = prepareKernel_ShermanMorrison(Psi, Y, uY, k, r1, opt.MultiBirth);

% ====== computing loss ======
uX = PsiB' * Psi;
for iclass = 1:k
    posLabel = uY(iclass);
    mask = xor(Y == posLabel, opt.MultiBirth);
    obj.simloss(1, iclass) = 0.5*sum(uX(iclass, mask).^2, "all");
    obj.disloss(1, iclass) = c * sum(max(0, 1+uX(iclass, ~mask)), "all");
end
obj.w2reg(1, :) = 0.5*r1*sum((PsiB - PsiA*V).^2);
obj.v2reg(1, :) = 0.5*r2*sum(V.^2);
obj.Preg(1)     = 0.5*mu*trace(PsiA'*KLK*PsiA);
obj.total(1, :) = obj.simloss(1, :) + obj.disloss(1, :) + obj.w2reg(1, :) + ...
                   obj.v2reg(1, :) + obj.Preg(1);
obj.tau(1, :)  = tau;
obj.dualobj(1) = obj.total(1, :)*tau;
obj.primalobj(1) = max(obj.total(1, :));

% test
if opt.verbose
    fprintf("weighted objective sum: %.6f\n", obj.dualobj(1));
end
% ====== computing loss ======


% iterative loop
for lp = 1:MAX_ITER
    % Note that KA is frequently used. Precompute it for acceleration
    PsiAV = PsiA*V;

    % update Psi*B
    for ii = 1:k
        Q = KQ{ii};
        h = -r1 * KMPsi{ii} * PsiAV(:, ii) - 1;

        % use coordinate descent for optmization
        alpha = cqpcd_box_(Q, h, 0, c, 1e-10);
        PsiB(:, ii) = r1 * PsiMPsi{ii} * PsiAV(:, ii) - KMPsi{ii}'*alpha;
    end

    % update V
    V = PsiA' * PsiB * r1 / (r1+r2);

    % update Psi*A
    PsiA = gpi(mu * KLK, r1 * PsiB * diag(tau) * V', PsiA);

    % ====== computing loss ======
    uX = PsiB' * Psi;
    for iclass = 1:k
        posLabel = uY(iclass);
        mask = xor(Y == posLabel, opt.MultiBirth);
        obj.simloss(1+lp, iclass) = 0.5*sum(uX(iclass, mask).^2, "all");
        obj.disloss(1+lp, iclass) = c * sum(max(0, 1+uX(iclass, ~mask)), "all");
    end
    obj.w2reg(1+lp, :)  = 0.5*r1*sum((PsiB - PsiA*V).^2);
    obj.v2reg(1+lp, :)  = 0.5*r2*sum(V.^2);
    obj.Preg(1+lp)      = mu*trace(PsiA'*KLK*PsiA);
    obj.total(1+lp, :)  = obj.simloss(1+lp, :) + obj.disloss(1+lp, :) + obj.w2reg(1+lp, :) + ...
                          obj.v2reg(1+lp, :) + obj.Preg(1+lp);
    obj.tau(1+lp, :)    = tau;
    obj.dualobj(1+lp)   = obj.total(1+lp, :)*tau;
    obj.primalobj(1+lp) = max(obj.total(1+lp, :));
    if opt.verbose
        fprintf("weighted objective sum: %.6f\n", obj.dualobj(1+lp));
    end
    % ====== computing loss ======

    % update tau
    Pv = PsiA*V;
    Pu = PsiB'*PsiA;
    for ll = 1:k
        temp = r1*(Pv(:, ll)*Pu(ll, :) - PsiB(:, ll) * V(:, ll)');
        Omega(:, ll) = temp(:);
    end
    pi_ = mu * (eye(n) - PsiA*PsiA')*KLK*PsiA;

    delta = quadprog(Omega'*Omega, Omega'*pi_(:) - gamma * obj.total(lp+1, :)', ...
                     [], [], ones(1,k), 1, zeros(k,1), [], [], qpoptions);
    if norm(Omega*delta + pi_(:)) > EPSILON
        tau = (1-BETA) * tau + BETA*delta; % momentum smoothing updating
        Grad = reshape(Omega*tau + pi_(:), [n, d]);
        PsiA = PsiA - LR * Grad;
        [uu, ~, vv] = svd(PsiA, "econ");
        PsiA = uu*vv';
    end
    if opt.verbose
        fprintf("P-gradient norm: %.6f\n", norm(Grad));
    end
    if norm(Grad) <= EPSILON ...
        % || abs(obj.primalobj(1+lp) - obj.dualobj(1+lp)) <= EPSILON
        fprintf("break.\n");
        break;
    end
end

npsvcpp.isLinearModel = isLinearKernel;
if isLinearKernel
    npsvcpp.P = PsiA;
    npsvcpp.U = PsiB;
    npsvcpp.V = V;
    npsvcpp.param = param;
    npsvcpp.multiBirth = opt.MultiBirth;
else
    npsvcpp.PsiA = PsiA;
    npsvcpp.PsiB = PsiB;
    npsvcpp.V = V;
    npsvcpp.Xtrain = X;
    npsvcpp.param = param;
    npsvcpp.A = Psi \ PsiA;
    npsvcpp.B = Psi \ PsiB;
    npsvcpp.multiBirth = opt.MultiBirth;
end

model.name  = "NPSVM++";
model.model = npsvcpp;
model.obj   = obj;

end

function [KQ, PsiMPsi, KMPsi] = prepareKernel_ShermanMorrison(Psi, Y, uY, k, r1, multiBirthOption)
% Sherman–Morrison inverse.
KQ      = cell(k,1);
PsiMPsi = cell(k,1);
KMPsi   = cell(k,1);
for l = 1:k
    posLabel = uY(l);
    mask = xor(Y == posLabel, multiBirthOption);

    Psi_p = Psi(:, mask);
    Psi_n = Psi(:, ~mask);

    n_p = size(Psi_p, 2);
    
    % intermediate result
    Ml = Psi_p / (Psi_p'*Psi_p + r1*eye(n_p)) * Psi_p';

    % store the result
    PsiMPsi{l} = (eye(size(Ml)) - Ml) / r1;
    KMPsi{l} = Psi_n' * PsiMPsi{l};             

    % make KQ symmetric
    KQ0 = KMPsi{l} * Psi_n;
    KQ{l} = (KQ0 + KQ0) / 2;
end
end