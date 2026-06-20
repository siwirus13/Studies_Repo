# Uniwersalny Makefile do asemblera (PIC16F877A)

# Znajduje wszystkie pliki .asm w obecnym folderze
SOURCES = $(wildcard *.asm)
# Zamienia w nazwach rozszerzenie .asm na .hex
TARGETS = $(SOURCES:.asm=.hex)

# Domyślny cel: wpisanie samego 'make' zbuduje wszystkie pliki .hex
all: $(TARGETS)

# Magiczna reguła: jak zrobić .hex z .asm
%.hex: %.asm
	gpasm -p p16f877a $<

# Komenda do sprzątania śmieci po kompilatorze
clean:
	rm -f *.hex *.lst *.err *.cod
