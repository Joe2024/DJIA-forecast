tic;clc;clear
%% program controllers (control parameters may be editted in this setion)
rp=xlsread('djia-85-2014-test303-2.xlsx');%real prices
ns=255;%number of starting day of prediction period(we predict the (k+1)'th day's price on k'th day)
nf=ns+4;%number of finishing day of prediction period
nin=[1 2  3];%number of inputs (shows how many days' prices are used to predict tomorrow's price)
ntr=[7 10 12  ];%number of data sets used in training the model
sp=[1];%smoothing parameter in smoothing spline method
nmf=[2];%number of membership functions
% imft=cellstr(['trimf   ';'trapmf  ';'gbellmf ';'gaussmf ';'gauss2mf';'pimf    ';'dsigmf  ';'psigmf  ']);%input membership function type
% omft=cellstr(['constant';'linear  ']);%output membership function type
imft=cellstr([ 'dsigmf' ]);%input membership function type
omft=cellstr(['constant']);%output membership function type
nepoch=[120];%number of epoches (we may replace it by error)
trerr=0;%training error goal
[ptps ptpr SS SR RS RR]=Pred_Prices(rp,ns,nf,nin,ntr,sp,nmf,imft,omft,nepoch,trerr);
save Pred_Prices
% -------------------------------------------------------------------------------------------------------
clc
[a b c]=size(ptpr);

for i1=1:c;
    for i2=1:b;
        for i3=1:a;
            
 d(i3,i2,i1)=abs(ptpr(i3,i2,i1)-rp(ns+i3))./rp(ns+i3);
 e(i3,i2,i1)=abs(ptps(i3,i2,i1)-rp(ns+i3))./rp(ns+i3);

        end
    end
end

PTPR=sum(d)./a
PTPS=sum(e)./a

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% mohasebeh Dpercent
[a b c]=size(ptpr);

for i1=1:c;
    for i2=1:b;
        for i3=1:a-1;
            if sign(ptpr(i3,i2,i1)-ptpr(i3+1,i2,i1))==sign(rp(ns+i3)-rp(ns+i3+1)) dpercent(i3,i2,i1)=1;
   else dpercent(i3,i2,i1)=0;
   end
 
        end
    end
end
for i1=1:c;
    for i2=1:b;
        Dpercent(1,i2,i1)=(sum(dpercent(:,i2,i1))./(a-1))*100
    end
end
    
[a b c]=size(ptps);

for i1=1:c;
    for i2=1:b;
        for i3=1:a-1;
            if sign(ptps(i3,i2,i1)-ptps(i3+1,i2,i1))==sign(rp(ns+i3)-rp(ns+i3+1)) dspercent(i3,i2,i1)=1;
   else dspercent(i3,i2,i1)=0;
   end
 
        end
    end
end
for i1=1:c;
    for i2=1:b;
        DSpercent(1,i2,i1)=(sum(dspercent(:,i2,i1))./(a-1))*100
    end
end
  
%---------------------------------------------------------------------------------------------------------


