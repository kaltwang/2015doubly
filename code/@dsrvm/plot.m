function [] = plot(obj)

s = get_statistics( obj, 'likelihood');
ax(1) = subplot(3,1,1);
plot(s(:,1));
title('delta logML');
ax(2) = subplot(3,1,2);
plot(s(:,10));
title('# Basis (M)');
ax(3) = subplot(3,1,3);
plot(s(:,11));
title('# Kernels (K)');
drawnow;
linkaxes([ax(3) ax(2) ax(1)],'x');

end