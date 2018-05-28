clear
n=2;m=3;Du=8;DJ=8;Dm=8;
dimx=n;dimu=m;
A=eye(n);
B=eye(m);

dJ=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for model NN
wm1=0.1*randn(Dm,(m+n))
wm2=0.1*(randn(Dm,n))'
% for ANN 
wau1=0.1*randn(Du,n)%inputhidden row Du colum n
wau2=0.1*(randn(Du,m))'%outputhidden row Du colum m

% for CNN
wc1=0.1*randn(DJ,n)%inputhidden row 8 colum 3
wc2=0.1*(randn(DJ,n))'%outputhidden row 1 colum 8

xpic=[];upic=[];dJpic=[];

a=0.3; %learning rate
x1=1.08;x2=0.66;

mm=1000;mmm=200;
for ii=1:mmm
    
u=0.1*rands(m,1);
x=0.1*rand(n,1);

for i=1:mm %training the model
   h=x(1);
   g=x(2);
sa=[h;g];
vz=[sa;u];
%for model
sc=vz;
xnexth1=wm1*sc;xnexth2=tansig(xnexth1);%%x->h1,x->h2
xnext=wm2*xnexth2;
ax=[h*100;0.0001388*h*100+0.9998*g*100];
bx=[-0.0018 -0.000698 -0.0003;-1.249e-007 0.02831 -2.082e-008];

xnextr=0.01*(ax+bx*u);% x->r
em=xnext-xnextr;

dwm2=a*xnexth2*em';%(8*1)*(1*2)=8*2
wm2=wm2-(dwm2)';
dwm1=a*((wm2'*em).*dtansig(xnexth2,xnexth1))*vz';
wm1=wm1-dwm1;
end
end %%训练完wm1,wm2
Ti=1000;
x=[x1;x2];
for i=1:Ti

h=x(1); %save x1
g=x(2); %save x2
sa=[h;g];%actionnet input
%%%%%%%%%%%compute J(t);
dJnh1=wc1*sa;dJnh2=tansig(dJnh1);
dJ=wc2*dJnh2;
%%%%%%%%%%%%%%%%%%%
uh1=wau1*sa;uh2=tansig(uh1);
u=wau2*uh2;

sc=[sa;u];%%%x,u,w,必须与model的输入顺序一致
xnexth1=wm1*sc;xnexth2=tansig(xnexth1);
xnext=wm2*xnexth2;

ax=[h*100;0.0001388*h*100+0.9998*g*100];
bx=[-0.0018 -0.000698 -0.0003;-1.249e-007 0.02831 -2.082e-008];

dJnh1=wc1*xnext;dJnh2=tansig(dJnh1);
dJn=wc2*dJnh2;


ecbefore=dJ;
if (x(1)>1.3||x(1)<1.1)
    reinf=-0.14;
else if (x(2)<0.60||x(2)>0.70)
        reinf=-0.42;
    else
        reinf=0;
    end
end
dudx=(wau2'.*[(1-uh2.*uh2) (1-uh2.*uh2) (1-uh2.*uh2)])'*wau1;
dxnextdx=(wm2'.*[(1-xnexth2.*xnexth2) (1-xnexth2.*xnexth2)])'*wm1(:,1:n);
dxnextdu=(wm2'.*[(1-xnexth2.*xnexth2) (1-xnexth2.*xnexth2)])'*wm1(:,(n+1):(n+m));
ecafter=(2*sa'*A+2*u'*B*dudx+dJn'*(dxnextdx+dxnextdu*dudx))';
ec=(ecafter-0.855*ecbefore+reinf);


%1 date critic network
dwc2=a*dJnh2*ec';%(8*1)*(1*2)=8*2
wc2=wc2-(dwc2)';

dwc1=a*(ec'*(wc2'.*[(1-dJnh2.*dJnh2) (1-dJnh2.*dJnh2)])')'*xnext';%根据Jenni Si的文章，i与i相乘因此应点乘，i与j相乘应叉乘。
wc1=wc1-dwc1;

 %2 date action u
  

dwau2=a*(ec'*((wc2'.*[(1-dJnh2.*dJnh2) (1-dJnh2.*dJnh2)])'*wc1)*((wm2'.*[(1-xnexth2.*xnexth2) (1-xnexth2.*xnexth2)])'*wm1(:,(n+1):(n+m))))'*uh2';
wau2=wau2-dwau2;

dwau1=a*(((ec'*((wc2'.*[(1-dJnh2.*dJnh2) (1-dJnh2.*dJnh2)])'*wc1)*((wm2'.*[(1-xnexth2.*xnexth2) (1-xnexth2.*xnexth2)])'*wm1(:,(n+1):(n+m)))))*(wau2'.*[(1-uh2.*uh2) (1-uh2.*uh2) (1-uh2.*uh2)])')'*sa';
wau1=wau1-dwau1;

dJpic=[dJpic dJ];
xpic=[xpic x];
upic=[upic u];
x=0.01*(ax+bx*u);
% sa;
N=rand(Ti,3);
Z(i,1)=x(1)*100;
M(i,1)=x(2)*100;
N(i,1)=u(1)';
N(i,2)=u(2)';
N(i,3)=u(3)';
H(i,1)=ecbefore(1);
G(i,1)=ecbefore(2);
end

subplot(4,1,1);
plot(Z);
subplot(4,1,2);
plot(M);
subplot(4,1,3);
plot(N);
subplot(4,1,4);
plot(G);