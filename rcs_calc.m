function [RCS] = rcs_calc(metasurface,m,n,lmd,d)

%lmd=double(lmd);
%d=double(d);
k=2*(pi)/lmd;

Ntheta=180; Nphi=180;
%syms theta phi
dtheta=pi/(2*Ntheta); dphi=2*pi/Nphi;
theta=linspace(0,pi/2,Ntheta);
phi=linspace(0,2*pi,Nphi);
%phi=0;
[THETA, PHI]=meshgrid(theta,phi);



% uncomment this section for a random metasurface

%elements=[0,1]; % 2 elements corresponding to 0, pi phase responses
%idx=randi(length(elements), m,n); % indexes to create NxN random metasurface with 2 elements
%metasurface_actual = pi.*metasurface;

%%%%%%%% get the metasurface based on the state returned by python code
% 1 ---> 0
% 2 ---> pi/4
% 3 ---> pi/2
% 4 ---> pi
reflect_phi = metasurface;
reflect_phi(reflect_phi==1)=0;
reflect_phi(reflect_phi==2)=pi/4;
reflect_phi(reflect_phi==3)=pi/2;
reflect_phi(reflect_phi==4)=pi;

%reflect_phi = pi.*metasurface; % resulting metasurface



%sequence = [0,1,0,1,0,1,0,1];
%metasurface = [1 0 1 0 1 %0 0 0; 0 0 1 0 0 0 1 0; 1 0 0 0 1 0 1 0; 1 0 0 1 0 0 1 1; 0 0 1 1 0 1 1 1; 0 0 0 0 1 0 0 0; 1 1 0 1 0 1 0 1; 0 1 1 0 0 1 0 0];
%metasurface = [1 0 1 0; 0 1 1 0; 1 0 0 1; 0 1 0 1];
%reflect_phi = pi.*metasurface;


F=0;
for q=1:n
   for p=1:m
   % if( mod(q,2)+ mod(l,2)==1 )
    % u=(unidrnd(2,1)-1)*180/180*pi;
  %  w=x(q,1);v=y(q,1);
  %   u=z5(q,1);
   %u=zp((p-1)*N+q,1);
   % if(mod(q,2)==0 )
    %    F2=exp(-(1j)*(k*d*sin(THETA).*((q-0.5)*cos(PHI)+(l-0.5)*sin(PHI))));
    % else
    
       F2=exp(-(1j)*(reflect_phi(q,p)+k*d*sin(THETA).*((q-0.5)*cos(PHI)+(p-0.5)*sin(PHI))));
       %F2=exp(-(1j)*((u)+k*d*sin(THETA).*((p-0.5)*cos(PHI)+(q-0.5)*sin(PHI))));    
  
       %     F2=exp(-(1j)*((u)+(k)*sin(THETA).*(w*cos(PHI)+v*sin(PHI))));
    % end
    F=F+F2;
  end
end

   
 % F=cos(THETA).*F;
  F3=abs(F.*F).*sin(THETA);
  F4=F.*F;
  
    %R=F(THETA,PHI);
    Ravg=sum(sum(F3*dtheta)*dphi);
    F5=F4/Ravg;
    
    % maximum directivity
    Rm=max(abs(F5(:)));
    
    RCS = (lmd^2)*Rm/(4*pi*m*n*d^2);
    
    
%F4=abs(F3).*abs(F3);
%Rm=max(abs(F(:)));
%Rm=max(abs(F4(:)));
%F=20*log10(abs(F));
   
%Rm1=max(abs(F4(:)/Ravg));

%F=F-max(F);
 %plot(theta,F)
 %axis([0 pi/2 -40 0])
%Rm1=abs(Ravg);

%% uncomment for plotting
%[x,y,z]=sph2cart(PHI,pi/2-THETA,abs(F5));
%figure(1);
%rotate3d on;
%mesh(x,y,z);

end