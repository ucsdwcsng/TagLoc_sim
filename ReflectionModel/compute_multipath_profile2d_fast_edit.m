function DP=compute_multipath_profile2d_fast_edit(h,theta_vals,d_vals,opt)
    
    freq_cent = median(opt.freq);
    const = 1j*2*pi/(3e8);
    const2 = 1j*2*pi*opt.ant_sep*freq_cent/(3e8);
    h = h.';
    d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
    temp = h*exp(d_rep);
    theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
    DP = exp(theta_rep)*(temp);
    
    
%                     DP(i,j)=DP(i,j)+h(k,l)*exp(1j*2*pi*(l*opt.ant_sep*sin(theta_vals(i))+d_vals(j))/opt.lambda(k));
%     DP=zeros(length(theta_vals),length(d_vals));
%     freqcomp = (1j*2*pi/(3e8)).*opt.freq;
%     for i=1:length(theta_vals)
%         for j = 1:length(d_vals)
%             temp = [1:size(h,2)].*(opt.ant_sep*sin(theta_vals(i)))+d_vals(j);
%             coeffecients = transpose(h).*exp(temp'*freqcomp);
%             DP(i,j)=sum(coeffecients(:));
%         end
%     end
end