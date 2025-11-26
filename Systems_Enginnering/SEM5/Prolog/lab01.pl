
p_sqrt(X, Y) :-
    Y is sqrt(X).

p_power(X, Y) :-
    Y is X ** 0.5.


log_n(X, Y) :-
    Y is log(X).

linear(A, B, X) :-
	A =/= 0,
	X is  -B/A.

quadratic(A, B, C, X1, X2) :-
    A =\= 0,
    Delta is B*B - 4*A*C,
    Delta > 0,                        
    X1 is (-B + sqrt(Delta)) / (2*A),
    X2 is (-B - sqrt(Delta)) / (2*A).

quadratic(A, B, C, X, X) :-
    A =\= 0,
    Delta is B*B - 4*A*C,
    Delta =:= 0,
    X is -B / (2*A).

quadratic(_, _, _, _, _) :-
	write('Brak rozwiazan rzeczywistych!'), nl, fail.

