% The mutuation logic won't let w&b flip signs, this is a temporary fix to
% see if training improves if I flip the near zero numbers
clc
format compact

% W = rand(2,3,'single')-.5
% A = abs(W);
% B = A < .1;
% B = B * -1;
% C = A >= .1;
% W = W.*B + W.*C

smallValue = .01;

W = weights.one(:,:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
weights.one(:,:,NN.winningIndex) = W;

W = weights.two(:,:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
weights.two(:,:,NN.winningIndex) = W;

% W = weights.three(:,:,NN.winningIndex);
% A = abs(W);
% B = A < smallValue;
% B = B * -1;
% C = A >= smallValue;
% W = W.*B + W.*C;
% weights.three(:,:,NN.winningIndex) = W;

W = weights.out(:,:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
weights.out(:,:,NN.winningIndex) = W;

W = biases.one(:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
biases.one(:,NN.winningIndex) = W;

W = biases.two(:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
biases.two(:,NN.winningIndex) = W;

% W = biases.three(:,NN.winningIndex);
% A = abs(W);
% B = A < smallValue;
% B = B * -1;
% C = A >= smallValue;
% W = W.*B + W.*C;
% biases.three(:,NN.winningIndex) = W;

W = biases.out(:,NN.winningIndex);
A = abs(W);
B = A < smallValue;
B = B * -1;
C = A >= smallValue;
W = W.*B + W.*C;
biases.out(:,NN.winningIndex) = W;

