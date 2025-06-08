## MATEUSZ PILECKI 279994


# Sposób kompilacji programu

```bash
gcc minmax.cpp -o <nazwa_programu> -lgsl -lgslcblas -lm
```
`<nazwa_programu>` może być dowolnym ciągiem znaków (przesłany skompilowany kod jest nazwany numerem indeksu)

# Sposób wywołania programu

```bash
./<nazwa_programu> <numer ip> <numer portu> <gracz> <nick> <poziom głębokości>
```

Chcąc wywołać skompilowany już kod należy użyć:

```bash
./279994 <numer ip> <numer portu> <gracz> <nick> <poziom głębokości>
```

# Narzędzia i biblioteki użyte w programie
Nic specjalnego - takie same biblioteki jak w przykładowym kodzie z pliku `labor3.zip`

# Informacje na temat heurystyki
Znajdują się w pliku minmax.cpp w języku angielskim jako komentarz na samej górze


