function [obj, selectedAction, deltaLogML] = update_ab( obj, G, t, ab)
% This is a modified version of SparseBayes.m by Mike Tipping. For the
% original code, see http://www.miketipping.com/

logML_before = get_likelihood( obj );

if ab == 'a'
    % alpha update
    BASIS = get_Phi(obj, G, true);
    Ui = obj.w_SU;
    Mu = obj.w;
    wv_idx = obj.w_idx;
    Alpha = obj.alpha;

    M_full = size(G,2);
    [M, ~] = get_MK(obj);
else
    % beta update
    BASIS = get_Psi(obj, G, true);
    Ui = obj.v_SU;
    Mu = obj.v;
    wv_idx = obj.v_idx;
    Alpha = obj.beta;

    M_full = size(G,3);
    [~, M] = get_MK(obj);
end

PHI = BASIS(:,wv_idx);
Used = find(wv_idx)';
beta = obj.sigma2inv;
BASIS_PHI = BASIS' * PHI;
BASIS_Targets = BASIS' * t;

% UPDATE STATISTICS
%
% Gaussian case simple: beta a scalar
% 
betaBASIS_PHI	= beta*BASIS_PHI;			
%
% The S_in calculation exploits the fact that BASIS is normalised,
%  i.e. sum(BASIS.^2,1)==ones(1,M_full)
%
% S_in		= beta - sum((betaBASIS_PHI*Ui).^2,2);
% for unnormalized BASIS:
S_in		= beta*sum(BASIS.^2,1)' - sum((betaBASIS_PHI*Ui).^2,2);
Q_in		= beta*(BASIS_Targets - BASIS_PHI*Mu);


%
S_out		= S_in;
Q_out		= Q_in;
%
% S,Q with that basis excluded: equations (23)
% 
S_out(Used)	= (Alpha .* S_in(Used)) ./ (Alpha - S_in(Used));
Q_out(Used)	= (Alpha .* Q_in(Used)) ./ (Alpha - S_in(Used));
%
% Pre-compute the "relevance factor" for ongoing convenience
% 
Factor		= (Q_out.*Q_out - S_out);


% CONSTANTS

% ACTION CODES
% Assign an integer code to the basic action types
ACTION_REESTIMATE	= 0;			
ACTION_ADD			= 1;
ACTION_DELETE		= -1;
%
% Some extra types
ACTION_TERMINATE		= 10;
%
% ACTION_NOISE_ONLY		= 11;
%
% ACTION_ALIGNMENT_SKIP	= 12;

% Any Q^2-S "relevance factor" less than this is considered to be zero
CONTROLS.ZeroFactor			= 1e-12;

% If the change in log-alpha for the best re-estimation is less than this,
% we consider termination
% CONTROLS.MinDeltaLogAlpha	= 1e-3;
CONTROLS.MinDeltaLogAlpha	= 1e-10;

% REDUNDANT BASIS
% Check for basis vector alignment/correlation redundancy
CONTROLS.BasisAlignmentTest		= false;

% ADD/DELETE
% - preferring addition where possible will probably make the algorithm a
% little slower and perhaps less "greedy"
% 
% - preferring deletion may make the model a little more sparse and the
% algorithm may run slightly quicker
% 
% Note: both these can be set to 'true' at the same time, in which case
% both take equal priority over re-estimation. 
CONTROLS.PriorityAddition	= obj.PriorityAddition;
CONTROLS.PriorityDeletion	= obj.PriorityDeletion;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DECISION PHASE
%
% Assess all potential actions
%

%
% Compute change in likelihood for all possible updates
% 
DeltaML		= zeros(M_full,1);
%
Action		= ACTION_REESTIMATE*ones(M_full,1); % Default
%
% 'Relevance Factor' (Q^S-S) values for basis functions in model
% 
UsedFactor	= Factor(Used);

%
% RE-ESTIMATION: must be a POSITIVE 'factor' and already IN the model
% 
iu		= UsedFactor>CONTROLS.ZeroFactor;
index		= Used(iu);
NewAlpha	= S_out(index).^2 ./ Factor(index);
Delta		= (1./NewAlpha - 1./Alpha(iu)); % Temp vector
%
% Quick computation of change in log-likelihood given all re-estimations
% 
DeltaML(index)	= (Delta.*(Q_in(index).^2) ./ ...
           (Delta.*S_in(index) + 1) - ...
           log(1 + S_in(index).*Delta))/2;

% 
% DELETION: if NEGATIVE factor and IN model
%
% But don't delete:
%		- any "free" basis functions (e.g. the "bias")
%		- if there is only one basis function (M=1)
% 
% (In practice, this latter event ought only to happen with the Gaussian
% likelihood when initial noise is too high. In that case, a later beta
% update should 'cure' this.)
% 
iu			= ~iu; 	% iu = UsedFactor <= CONTROLS.ZeroFactor
index			= Used(iu);
anyToDelete	= ~isempty(index) && M>1;
%
if anyToDelete
%
% Quick computation of change in log-likelihood given all deletions
% 
DeltaML(index)	= -(Q_out(index).^2 ./ (S_out(index) + Alpha(iu)) - ...
            log(1 + S_out(index) ./ Alpha(iu)))/2;
Action(index)	= ACTION_DELETE;
% Note: if M==1, DeltaML will be left as zero, which is fine
end

% 
% ADDITION: must be a POSITIVE factor and OUT of the model
% 
% Find ALL good factors ...
GoodFactor		= Factor>CONTROLS.ZeroFactor;
% ... then mask out those already in model
GoodFactor(Used)	= 0;		
% ... and then mask out any that are aligned with those in the model
if CONTROLS.BasisAlignmentTest
GoodFactor(Aligned_out)	= 0;
end
%
index			= find(GoodFactor);
anyToAdd		= ~isempty(index);
if anyToAdd
%
% Quick computation of change in log-likelihood given all additions
% 
quot			= Q_in(index).^2 ./ S_in(index);
DeltaML(index)	= (quot - 1 - log(quot))/2;
Action(index)	= ACTION_ADD;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Post-process action results to take account of preferences

% Ensure that nothing happens with "free basis" functions
% 
% DeltaML(OPTIONS.freeBasis)	= 0;

% If we prefer ADD or DELETE actions over RE-ESTIMATION
% 
if (anyToAdd && CONTROLS.PriorityAddition) || ...
  (anyToDelete && CONTROLS.PriorityDeletion)
% We won't perform re-estimation this iteration, which we achieve by
% zero-ing out the delta
DeltaML(Action==ACTION_REESTIMATE)	= 0;
% Furthermore, we should enforce ADD if preferred and DELETE is not
% - and vice-versa
if (anyToAdd && CONTROLS.PriorityAddition && ~CONTROLS.PriorityDeletion)
  DeltaML(Action==ACTION_DELETE)	= 0;
end
% if (anyToDelete && CONTROLS.PriorityDeletion && ~CONTROLS.PriorityAddition)
%   DeltaML(Action==ACTION_ADD)		= 0;
% end
if (anyToDelete && CONTROLS.PriorityDeletion)
  DeltaML(Action==ACTION_ADD)		= 0;
end
end

% Finally...we choose the action that results 
% in the greatest change in likelihood
% 
[deltaLogMarginal, nu]	= max(DeltaML);
selectedAction		= Action(nu);
anyWorthwhileAction	= deltaLogMarginal>0;
%
% We need to note if basis nu is already in the model, and if so,
% find its interior index, denoted by "j"
%
if selectedAction==ACTION_REESTIMATE || selectedAction==ACTION_DELETE	
j		= find(Used==nu);
end
%
% Get the individual basis vector for update and compute its optimal alpha,
% according to equation (20): alpha = S_out^2 / (Q_out^2 - S_out) 
%
% Phi		= BASIS(:,nu);
newAlpha	= S_out(nu)^2 / Factor(nu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TERMINATION CONDITIONS
%
% Propose to terminate if:
% 
% 1.	there is no worthwhile (likelihood-increasing) action, OR
% 
% 2a.	the best action is an ACTION_REESTIMATE but this would only lead to
%		an infinitesimal alpha change, AND
% 2b.	at the same time there are no potential awaiting deletions
% 
if ~anyWorthwhileAction || ...
(selectedAction==ACTION_REESTIMATE && ...
 abs(log(newAlpha) - log(Alpha(j)))<CONTROLS.MinDeltaLogAlpha && ...
 ~anyToDelete)
%
selectedAction	= ACTION_TERMINATE;
%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ACTION PHASE
%
% Implement above decision
%
  
switch selectedAction
    case ACTION_REESTIMATE
        Alpha(j)	= newAlpha;

    case ACTION_ADD
        Alpha_old = Alpha;
        Alpha	= [Alpha ; newAlpha];
        Used		= [Used; nu];

    case ACTION_DELETE
        Alpha(j)	= [];
        Used(j)		= [];
end

M = length(Used);

if selectedAction == ACTION_ADD
    % we need to change the order of Alpha to be able to use logical
    % indexing
    Alpha_new = zeros(M,1);
    sortUsed = sort(Used);
    Alpha_new(wv_idx(sortUsed)) = Alpha_old;
    Alpha_new(~wv_idx(sortUsed)) = newAlpha;

    Alpha = Alpha_new;
end

wv_idx = false(1,M_full);
wv_idx(Used) = true;

if ab == 'a'
    obj.alpha = Alpha;
    obj.w_idx = wv_idx;

    % these values will be set at the next w update step
    obj.w = zeros(M,1);
    obj.w_S = zeros(M,M);
    obj.w_logdetSOver2 = 0;
    obj.w_SU = zeros(M,M);
else
    obj.beta = Alpha;
    obj.v_idx = wv_idx;

    % these values will be set at the next v update step
    obj.v = zeros(M,1);
    obj.v_S = zeros(M,M);
    obj.v_logdetSOver2 = 0;
    obj.v_SU = zeros(M,M);
end

if selectedAction == ACTION_ADD || selectedAction == ACTION_DELETE
    % update of GtG needed
    % ToDo: incremental update of GtG
    obj = update_GtG( obj, G);
end

if ab == 'a'
    wv = 'w';
else
    wv = 'v';
end
[obj] = update_wv( obj, G, t, wv);
logML_after = get_likelihood( obj );
deltaLogML = logML_after - logML_before;

obj.action = selectedAction;
obj.update = wv;

end