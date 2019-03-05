%Bruce Denby.  Paris, 7 february 2005.  
%modelled on write_gsm_worrell_new. writes out an audio file to see how the net did

clear

%get the training file to retrieve the means and sigs
	load('prototrain')
	coker=targs';
	mogo=mean(coker);
	sogo=std(coker);
	for i=1:length(coker),
		coaxo(i,:)=(coker(i,:)-mogo)./sogo;
	end

%load the saved nn file to get the c nn output matrix
load('LSP_net_chekit_bfg','c');

%kluge!!!  replace coefficients 8, 9, and 11 by their nearly constant values for all frames...
c(12,:)=0;
c(11,:)=0.4;
c(10,:)=0;
c(9,:)=-0.4;
c(8,:)=0.6;
c(1,:)=0;
c(3,:)=0;

%unnormalize the nn output by multiplying by std and adding mean.
for i=1:length(c),
	outcoeff(:,i) = c(:,i).*sogo' + mogo';
end	

%coeffs have to be integers
%wcoeff=round(outcoeff);

dfen=11025/30;

numsen=zeros(4);
numsen(1)=6;
numsen(2)=6;
numsen(3)=9;
numsen(4)=9;

nframe=zeros(4,9);

%rain 2
nframe(1,1) = 153;
nframe(1,2) = 113;
nframe(1,3) = 227;
nframe(1,4) = 149;
nframe(1,5) =  79;
nframe(1,6) = 192;

%now rain1
nframe(2,1) = 160;    
nframe(2,2) = 107;
nframe(2,3) = 235;
nframe(2,4) = 150;
nframe(2,5) =  84;
nframe(2,6) = 187;

%then grand2
nframe(3,1) =  61;   
nframe(3,2) = 158;
nframe(3,3) = 172;
nframe(3,4) = 179;
nframe(3,5) = 124;
nframe(3,6) = 125;
nframe(3,7) = 213;
nframe(3,8) = 199;
nframe(3,9) =  90;

%and finally grand1 
nframe(4,1) =  76;
nframe(4,2) = 154;
nframe(4,3) = 165;
nframe(4,4) = 195;
nframe(4,5) = 133;
nframe(4,6) = 148;
nframe(4,7) = 201;
nframe(4,8) = 203;
nframe(4,9) =  89;

ptr=0;

for par=1:4
   for sentence=1:numsen(par)
      clear lpccoef brak nbfen;
      for k=1:nframe(par,sentence)-1
			lpccoef = polystab(lsf2poly(outcoeff(:,ptr+k)));
			ind = max((floor((2*k-1)*dfen/2)-185),1):floor((2*k-1)*dfen/2)+185;
         brak(ind) = filter(1,lpccoef,randn(1,length(ind)));
      end;
      wavwrite(brak./(1.01*max(abs(brak))),11025,['nnout_2020_bfg_zerout4' num2str(par) '_' num2str(sentence)]);
		ptr = ptr + nframe(par,sentence)-1;
	end;
    
    `


