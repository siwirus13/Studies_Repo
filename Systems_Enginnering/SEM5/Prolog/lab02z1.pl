
% ZADANIE 1 – MIESZKAŃCY

osoba(ania, 20).
osoba(bartek, 25).
osoba(celina, 30).
osoba(darek, 40).
osoba(ewa, 35).

mieszka_nad(ania, bartek).
mieszka_nad(bartek, celina).
mieszka_nad(celina, darek).
mieszka_nad(ewa, ania).

mieszka_pod(X, Y) :- mieszka_nad(Y, X).

mieszka_wyzej(X, Y) :- mieszka_nad(X, Y).
mieszka_wyzej(X, Y) :-
    mieszka_nad(X, Z),
    mieszka_wyzej(Z, Y).

mieszka_nizej(X, Y) :- mieszka_pod(X, Y).
mieszka_nizej(X, Y) :-
    mieszka_pod(X, Z),
    mieszka_nizej(Z, Y).

mieszka_najwyzej(X) :-
    \+ mieszka_nad(_, X).

mieszka_najnizej(X) :-
    \+ mieszka_pod(_, X).


jest_starsza(X, Y) :-
    osoba(X, WX),
    osoba(Y, WY),
    WX > WY.

jest_mlodsza(X, Y) :-
    osoba(X, WX),
    osoba(Y, WY),
    WX < WY.

jest_najstarsza(X) :-
    osoba(X, WX),
    \+ (osoba(_, WY), WY > WX).

jest_najmlodsza(X) :-
    osoba(X, WX),
    \+ (osoba(_, WY), WY < WX).
