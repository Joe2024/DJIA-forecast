tic;clear;clc
%% program controllers (control parameters may be editted in this setion)
rp=xlsread('goldquarterly.xlsx');%real prices
ns=44;%number of starting day of prediction period(we predict the (k+1)'th day's price on k'th day)
nf=53;%number of finishing day of prediction period
nin=[ 2 3 ];%number of inputs (shows how many days' prices are used to predict tomorrow's price)
ntr=[5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ];%number of data sets used in training the model
sp=[1 ];%smoothing parameter in smoothing spline method
nmf=[2];%number of membership functions
% imft=cellstr(['trimf   ';'trapmf  ';'gbellmf ';'gaussmf ';'gauss2mf';'pimf    ';'dsigmf  ';'psigmf  ']);%input membership function type
% omft=cellstr(['constant';'linear  ']);%output membership function type
imft=cellstr(['gbellmf']);%input membership function type
omft=cellstr(['constant']);%output membership function type
nepoch=[100];%number of epoches (we may replace it by error)
trerr=0;%training error goal
%% output description

%ptps and ptpr represent predicted tomorrow's prices. ptps uses smoothed
%prices to predict tomorrow's prices, whie ptpr uses real prices to predict
%tomorrow's prices.

%SS, SR, RS, and RR represent the percentage of correct predictions. Their
%indices refer to day number, nin, ntr, sp, nmf, imft, omft, and nepoch
%respectively.

%Note that:

%ptps


%SS uses ***Smoothed*** data as the inputs to the ANFIS model, and
%evaluates the measures the increse or decrease of the predicted price with
%respect to today's ***Smoothed*** price.

%SR uses ***Smoothed*** data as the inputs to the ANFIS model, and
%evaluates the measures the increse or decrease of the predicted price with
%respect to today's ***Real*** price.

%RS uses ***Real*** data as the inputs to the ANFIS model, and evaluates
%the measures the increse or decrease of the predicted price with respect
%to today's ***Smoothed*** price.

%RR uses ***Real*** data as the inputs to the ANFIS model, and evaluates
%the measures the increse or decrease of the predicted price with respect
%to today's ***Real*** price.
%% calling the prediction function
[ptps ptpr SS SR RS RR]=Pred_Prices(rp,ns,nf,nin,ntr,sp,nmf,imft,omft,nepoch,trerr);
save Pred_Prices