function result = NPSVCPP_predict(Xtest, Ytest, model, varargin)
% 

p = inputParser;
p.KeepUnmatched(true);
p.addParameter("manualKernel", false, ...
    @(x) assert(islogical(x), "manualKernel option should be logical."));
parse(p, varargin{:});
opt = p.Results;

npsvcpp = model.model;

% check linearity.
if npsvcpp.isLinearModel
    P = npsvcpp.P;
    U = npsvcpp.U;
    V = npsvcpp.V;
    wnorm2 = sum((U - P*V).^2);
    uNorm = sqrt(wnorm2 + sum(V.^2));
    y = Xtest * U;
else
    PsiA = npsvcpp.PsiA;
    PsiB = npsvcpp.PsiB;
    B = npsvcpp.B;
    V = npsvcpp.V;
    Xtrain = npsvcpp.Xtrain;
    param = npsvcpp.param;
    wnorm2 = sum((PsiB - PsiA*V).^2);
    uNorm = sqrt(wnorm2 + sum(V.^2));
    if opt.manualKernel
        y = Xtest * B;
    else
        y = constructKernel(Xtest, Xtrain, param.kernel) * B;
    end
end

% score denotes the distance to the hyperplane
score = abs(y) ./ uNorm;

% optional multi-class strategy
if ~npsvcpp.multiBirth
    [~, mid] = min(score, [], 2);
else
    [~, mid] = max(score, [], 2);
end

pred = npsvcpp.uY(mid);
if ~isempty(Ytest)
    accu = 100 * sum(pred(:) == Ytest(:)) / numel(Ytest);
else
    accu = nan;
end

result.eval.accuracy = accu;
result.pred = pred;

end