
% ============================================
% ZADANIE 2 – DRZEWO GENEALOGICZNE
% ============================================

mezczyzna(jan).
mezczyzna(pawel).
mezczyzna(marek).
kobieta(anna).
kobieta(kasia).
kobieta(ola).

rodzic(jan, pawel).
rodzic(jan, kasia).
rodzic(anna, pawel).
rodzic(anna, kasia).
rodzic(pawel, marek).
rodzic(ola, marek).

wiek(jan, 70).
wiek(anna, 68).
wiek(pawel, 40).
wiek(ola, 38).
wiek(marek, 10).
wiek(kasia, 35).


ojciec(X, Y) :-
    mezczyzna(X),
    rodzic(X, Y).

matka(X, Y) :-
    kobieta(X),
    rodzic(X, Y).

dziecko(X, Y) :-
    rodzic(Y, X).

syn(X, Y) :-
    mezczyzna(X),
    dziecko(X, Y).

corka(X, Y) :-
    kobieta(X),
    dziecko(X, Y).

dziadek(X, Y) :-
    mezczyzna(X),
    rodzic(X, Z),
    rodzic(Z, Y).

babcia(X, Y) :-
    kobieta(X),
    rodzic(X, Z),
    rodzic(Z, Y).

brat(X, Y) :-
    mezczyzna(X),
    rodzic(Z, X),
    rodzic(Z, Y),
    X \= Y.

siostra(X, Y) :-
    kobieta(X),
    rodzic(Z, X),
    rodzic(Z, Y),
    X \= Y.

wujek(X, Y) :-
    mezczyzna(X),
    rodzic(Z, Y),
    brat(X, Z).

ciocia(X, Y) :-
    kobieta(X),
    rodzic(Z, Y),
    siostra(X, Z).

kuzyn(X, Y) :-
    mezczyzna(X),
    rodzic(A, X),
    rodzic(B, Y),
    brat(A, B).

kuzynka(X, Y) :-
    kobieta(X),
    rodzic(A, X),
    rodzic(B, Y),
    brat(A, B).


starszy(X, Y) :-
    wiek(X, WX),
    wiek(Y, WY),
    WX > WY.

mlodszy(X, Y) :-
    wiek(X, WX),
    wiek(Y, WY),
    WX < WY.

najstarszy(X) :-
    wiek(X, WX),
    \+ (wiek(_, WY), WY > WX).

najmlodszy(X) :-
    wiek(X, WX),
    \+ (wiek(_, WY), WY < WX).


przodek(X, Y) :-
    rodzic(X, Y).
przodek(X, Y) :-
    rodzic(X, Z),
    przodek(Z, Y).

potomek(X, Y) :-
    przodek(Y, X).

glowa_rodu(X) :-
    mezczyzna(X),
    \+ rodzic(_, X).  % nie ma rodzica
